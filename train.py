# -*- coding: utf-8 -*-
"""
一个使用Hugging Face TRL库对Gemma模型进行LoRA微调的完整脚本。
流程包括：环境设置、数据准备、模型训练、推理验证以及模型合并。
只需修改MODEL_ID，其他路径自动生成。

安装依赖：
pip install pytorch,bitsandbytes,datasets,huggingface-hub,peft,pillow,torch,transformers,trl
    
"""

import os
import json
import shutil
import torch
from datasets import load_dataset, Dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
import logging

# --- 全局配置 ---
# 设置日志级别，方便调试
logging.basicConfig(level=logging.INFO)

# --- 第1部分: 环境设置与模型加载 ---
logging.info("--- Part 1: 环境设置与模型加载 ---")

# 基础目录设置
BASE_DIR = "."

# 数据集文件
DATASET_FILE = "dataset.jsonl"

# 定义模型ID - 只需要修改这一行！
MODEL_ID = "google/gemma-3-270m-it"

# 定义设备（cpu: None, gpu: "cuda"）
DEVICE = None

# 自动生成模型相关的路径和名称
def generate_paths_from_model_id(model_id):
    """根据model_id自动生成相关路径"""
    # 提取模型名称，去掉组织前缀，处理特殊字符
    model_name = model_id.split('/')[-1].replace('-', '_')
    
    ollama_name = f"{model_name}_lora_merged"
    adapter_path = os.path.join(BASE_DIR, f"{model_name}_lora_adapter")
    merged_path = os.path.join(BASE_DIR, f"{model_name}_lora_merged")
    
    return ollama_name, adapter_path, merged_path

OLLAMA_MODEL_NAME, ADAPTER_PATH, MERGED_MODEL_PATH = generate_paths_from_model_id(MODEL_ID)

logging.info(f"模型ID: {MODEL_ID}")
logging.info(f"Ollama模型名: {OLLAMA_MODEL_NAME}")
logging.info(f"适配器路径: {ADAPTER_PATH}")
logging.info(f"合并模型路径: {MERGED_MODEL_PATH}")

# 登录Hugging Face (推荐使用huggingface-cli login命令，而不是在代码中硬编码token)
# 如果环境中已经设置了HF_TOKEN，则无需此行
# login(token='YOUR_HF_TOKEN')

# 加载分词器 (Tokenizer)
logging.info(f"正在加载模型 '{MODEL_ID}' 的分词器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# Gemma模型可能没有默认的填充令牌，需要手动设置
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 配置量化参数
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 启用4位量化
    bnb_4bit_use_double_quant=True,       # 是否使用双重量化
    bnb_4bit_quant_type="nf4",             # 量化类型，nf4是一种归一化浮点4位格式
    bnb_4bit_compute_dtype=torch.float16 # 计算时使用的dtype，加速且精度适中
)

# 加载基础模型
logging.info(f"正在加载基础模型 '{MODEL_ID}'...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,  # 传入量化配置
    device_map=DEVICE,  # 自动选择设备 (CPU/GPU)
    dtype=torch.float16, # 使用bfloat16以节省显存并加速
    trust_remote_code=True,
    low_cpu_mem_usage=True  # 可选，减小CPU内存占用
)

# 准备模型进行量化训练
model = prepare_model_for_kbit_training(model)

logging.info(f"✅ {MODEL_ID} 模型和分词器已成功加载！")
logging.info(f"模型显存占用: {model.get_memory_footprint() / 1e9:.2f} GB")


# --- 第2部分: 训练数据准备 ---
logging.info("\n--- Part 2: 训练数据准备 ---")

# Alpaca格式的训练样本
training_examples = [
    {"messages": [
        {"role": "user", "content": "分析一下美团最新的财报，重点看外卖业务的增长驱动力是什么？"},
        {"role": "assistant", "content": "根据最新的财报，美团外卖业务的增长主要由订单频率的增加和客单价的微升共同驱动。尤其是在三线及以下城市，用户渗透率的提升带来了显著的订单量增长。"}
    ]},
    {"messages": [
        {"role": "user", "content": "今天港股大跌，美团的股价表现如何，有什么需要关注的技术指标吗？"},
        {"role": "assistant", "content": "在今天恒生指数整体下挫2.5%的背景下，美团股价下跌了4.1%，风险显著放大。虽然RSI指标已进入超卖区间，可能存在技术性反弹，但更应警惕的是，股价已跌破120港元这一关键长期支撑位，若无法迅速收复，则下降通道可能被进一步确认，后续仍有下行压力。"}
    ]}
]

# 将样本逐行写入JSONL文件
with open(DATASET_FILE, 'w', encoding='utf-8') as f:
    for example in training_examples:
        f.write(json.dumps(example, ensure_ascii=False) + '\n')

print(f"文本数据文件 '{DATASET_FILE}' 创建成功！")

# 使用datasets库加载数据
text_dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
print("✅ 数据集加载成功！")
print("数据集内容预览:", text_dataset[0])


# --- 第3部分: 核心微调流程 ---
logging.info("\n--- Part 3: 核心微调流程 ---")

# 1. 配置LoRA
lora_config = LoraConfig(
    r=16,                      # LoRA的秩，增加到16以获得可能更好的性能
    lora_alpha=32,             # 通常设为r的两倍
    lora_dropout=0.05,         # Dropout比率，防止过拟合
    target_modules="all-linear",# 将LoRA应用到所有线性层
    task_type="CAUSAL_LM",     # 任务类型为因果语言模型
)

# 2. 配置训练参数
training_args = SFTConfig( # <--- 关键修复：使用 SFTConfig
    output_dir=os.path.join(BASE_DIR, "training_checkpoints"), # 训练检查点输出目录
    per_device_train_batch_size=2, # 每个GPU的批次大小
    gradient_accumulation_steps=4, # 梯度累积步数，有效批次大小 = 2 * 4 = 8
    learning_rate=2e-4,            # 学习率
    num_train_epochs=3,            # 训练轮次
    logging_steps=1,               # 每1步记录一次日志
    # optim="paged_adamw_8bit",      # 使用8位分页优化器以节省显存
    bf16=False,                    # 启用BF16混合精度训练 (如果GPU支持)
    report_to="none",              # 关闭外部报告工具 (如wandb)
    max_length=1024,           # <--- 正确位置：在这里设置最大序列长度
)

# 3. 创建SFTTrainer实例
trainer = SFTTrainer(
    model=model,
    train_dataset=text_dataset,
    peft_config=lora_config,
    args=training_args,
    processing_class=tokenizer, # <--- 关键修复：使用 'processing_class' 替代 'tokenizer'
)

# 4. 开始训练
logging.info("🚀 开始微调...")
trainer.train()
logging.info("🎉 微调完成！")

# 5. 保存训练好的LoRA适配器
trainer.save_model(ADAPTER_PATH)
logging.info(f"✅ LoRA适配器已保存至: {ADAPTER_PATH}")


# --- 第4部分: 推理与验证 ---
logging.info("\n--- Part 4: 推理与验证 ---")

# 清理GPU缓存
del trainer
del model
if DEVICE == "cuda":
    torch.cuda.empty_cache()

# 1. 重新加载基础模型和LoRA适配器进行推理
logging.info("为推理重新加载模型和适配器...")
base_model_for_inference = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,  # 传入量化配置
    device_map=DEVICE,
    dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True  # 可选，减小CPU内存占用
)

# 准备模型进行量化训练
base_model_for_inference = prepare_model_for_kbit_training(base_model_for_inference)

model_with_adapter = PeftModel.from_pretrained(base_model_for_inference, ADAPTER_PATH)
logging.info("✅ LoRA适配器加载成功，模型已准备好对话！")


def chat_with_finetuned_model(question: str):
    """使用微调后的模型进行单轮对话的函数"""
    logging.info("\n模型正在生成回答...")
    
    # 构建与训练时完全相同的聊天模板
    messages = [{"role": "user", "content": question}]

    # 应用模板并进行编码
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model_with_adapter.device)

    # 生成回答
    outputs = model_with_adapter.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )

    # 解码并打印结果
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("\n--- 模型回答 ---")
    print(response_text)
    print("------------------")

# 2. 开始测试
chat_with_finetuned_model("分析一下近期恒生科技指数的趋势")
chat_with_finetuned_model("美团最新的财报怎么样？")


# --- 第5部分: 合并并保存模型 ---
logging.info("\n--- Part 5: 合并并保存模型 ---")

# 1. 合并适配器权重到基础模型
logging.info("正在合并LoRA适配器...")
merged_model = model_with_adapter.merge_and_unload()
logging.info("✅ 模型合并完成")

# 2. 保存合并后的完整模型和分词器
logging.info(f"正在保存合并后的模型至: {MERGED_MODEL_PATH}")
merged_model.save_pretrained(MERGED_MODEL_PATH)
tokenizer.save_pretrained(MERGED_MODEL_PATH)
logging.info("✅ 合并后的模型和分词器已成功保存！")

exit()
# --- (可选) 第6部分: 创建Ollama模型文件 ---
logging.info("\n--- (Optional) Part 6: 创建Ollama模型文件 ---")

try:
    # 检查原始tokenizer.model文件是否存在
    # 注意：这部分依赖于Hugging Face的缓存结构，可能不稳定
    hub_cache_path = os.path.expanduser("~/.cache/huggingface/hub")
    snapshot_path_pattern = os.path.join(hub_cache_path, f"models--{MODEL_ID.replace('/', '--')}", "snapshots")
    
    if os.path.exists(snapshot_path_pattern):
        # 获取最新的snapshot
        latest_snapshot = sorted(os.listdir(snapshot_path_pattern))[-1]
        source_tokenizer_model = os.path.join(snapshot_path_pattern, latest_snapshot, "tokenizer.model")
        
        if os.path.exists(source_tokenizer_model):
            destination_tokenizer_model = os.path.join(MERGED_MODEL_PATH, "tokenizer.model")
            shutil.copy(source_tokenizer_model, destination_tokenizer_model)
            logging.info(f"已从缓存复制 tokenizer.model 到 {MERGED_MODEL_PATH}")

            # 创建Modelfile
            modelfile_content = f"FROM {MERGED_MODEL_PATH}"
            modelfile_path = os.path.join(BASE_DIR, "Modelfile")
            with open(modelfile_path, "w") as f:
                f.write(modelfile_content)
            
            # 创建并运行Ollama模型
            logging.info(f"正在创建Ollama模型: {OLLAMA_MODEL_NAME}")
            os.system(f"ollama create {OLLAMA_MODEL_NAME} -f {modelfile_path}")
            logging.info(f"✅ Ollama模型创建成功！请使用 `ollama run {OLLAMA_MODEL_NAME}` 运行。")
        else:
            logging.warning("在Hugging Face缓存中未找到 tokenizer.model，跳过Ollama创建。")
    else:
        logging.warning("未找到Hugging Face模型缓存目录，跳过Ollama创建。")

except Exception as e:
    logging.error(f"创建Ollama模型时出错: {e}")

logging.info(f"\n🎉 微调完成！生成的文件:")
logging.info(f"- LoRA适配器: {ADAPTER_PATH}")
logging.info(f"- 合并模型: {MERGED_MODEL_PATH}")
logging.info(f"- Ollama模型: {OLLAMA_MODEL_NAME}")
