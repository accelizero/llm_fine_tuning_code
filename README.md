# **LoRA 微调端到端脚本**

这是一个功能完整、开箱即用的Python脚本，用于使用Hugging Face的TRL（Transformer Reinforcement Learning）库对Google的Gemma系列模型进行LoRA（Low-Rank Adaptation）微调。

该脚本覆盖了从环境设置到模型部署的整个生命周期，旨在最大限度地简化微调流程。您**只需修改一行代码（MODEL\_ID）**，即可对不同的Gemma模型（或其他兼容的Hugging Face模型）进行微调。

## **🚀 主要特性**

* **一站式体验**: 包含数据准备、模型训练、推理验证、模型合并以及（可选的）Ollama模型创建的完整工作流。  
* **极简配置**: 只需更改MODEL\_ID变量，即可轻松切换基础模型。所有相关路径和名称都会自动生成。  
* **高效微调**: 采用PEFT库的LoRA方法，显著降低了微调所需的计算资源和时间。  
* **清晰的流程**: 代码分为六个逻辑部分，配有详细的日志输出，方便理解和调试。  
* **可扩展性**: 您可以轻松替换脚本中的示例数据为您自己的数据集。  
* **本地部署**: 包含将微调后模型打包为Ollama格式的可选步骤，方便在本地快速部署和运行。

## **🔧 环境准备**

在运行脚本之前，请确保您已安装所有必需的Python库。

1. 安装依赖:  
   打开终端并运行以下命令：  
   pip install torch bitsandbytes datasets huggingface-hub peft pillow transformers trl accelerate

   *注意：accelerate 库对于 SFTTrainer 是必需的。*  
2. Hugging Face 登录 (推荐):  
   为了能够下载需要身份验证的模型或上传您自己的模型，建议登录您的Hugging Face账户：  
   huggingface-cli login

   您需要输入一个具有read权限的Hugging Face访问令牌。

## **🏃‍♀️ 快速开始**

1. 配置模型:  
   打开 train.py 脚本，找到以下行：  
   \# 定义模型ID \- 只需要修改这一行！  
   MODEL_ID = "google/gemma-3-270m-it"

   将 "google/gemma-3-270m-it" 替换为您想要微调的任何Gemma模型ID（例如 google/gemma-7b-it）。  
2. 准备数据 (可选):  
   脚本内置了两条关于金融分析的示例数据。您可以直接运行，或者修改 training\_examples 列表来使用您自己的数据。请确保您的数据遵循Alpaca的对话格式：  
   \[  
       {"messages": \[  
           {"role": "user", "content": "你的问题..."},  
           {"role": "assistant", "content": "期望的回答..."}  
       \]},  
       \# ... 更多样本  
   \]

3. 运行脚本:  
   在您的终端中，直接运行Python脚本：  
   python train.py

4. 查看结果:  
   脚本执行完毕后，将在当前目录下生成以下文件夹：  
   * **\<model\_name\>\_lora\_adapter/**: 训练好的LoRA适配器权重。  
   * **\<model\_name\>\_lora\_merged/**: 与基础模型合并后的完整模型，可直接用于部署。  
   * **training\_checkpoints/**: 训练过程中保存的检查点。

## **📜 脚本流程详解**

脚本的执行过程被清晰地划分为几个部分：

* **第1部分: 环境设置与模型加载**  
  * 根据 MODEL\_ID 自动生成所有必要的路径和名称。  
  * 加载指定模型的分词器（Tokenizer）和基础模型。  
* **第2部分: 训练数据准备**  
  * 定义或加载您的训练数据。  
  * 将数据格式化为 .jsonl 文件并使用 datasets 库加载。  
* **第3部分: 核心微调流程**  
  * 配置 LoraConfig (如 r, lora\_alpha 等)。  
  * 配置 SFTConfig (如学习率、批次大小、训练轮次等)。  
  * 实例化 SFTTrainer 并启动训练。  
  * 训练完成后，保存LoRA适配器。  
* **第4部分: 推理与验证**  
  * 清理GPU显存。  
  * 重新加载基础模型，并将训练好的LoRA适配器附加到模型上。  
  * 使用 chat\_with\_finetuned\_model 函数测试微调后模型的效果。  
* **第5部分: 合并并保存模型**  
  * 将LoRA适配器的权重合并到基础模型中。  
  * 将合并后的完整模型保存到本地，以便后续使用或分发。  
* **第6部分: (可选) 创建Ollama模型文件**  
  * 这是一个可选步骤，用于在本地创建Ollama模型。  
  * 它会自动生成一个 Modelfile 并调用 ollama create 命令。  
  * *注意：此步骤需要您本地已安装并运行Ollama。*

## **💡 自定义与提示**

* **GPU显存**: 如果遇到显存不足（Out of Memory）的错误，可以尝试以下方法：  
  * 减小 SFTConfig 中的 per\_device\_train\_batch\_size。  
  * 减小 LoraConfig 中的 r (秩)。  
  * 在 SFTConfig 中启用 fp16=True 或 bf16=True (如果您的GPU支持)。  
* **训练数据**: 要使用您自己的数据集，只需替换 training\_examples 变量的内容，或修改代码以从文件中加载数据（例如CSV或JSON文件）。  
* **模型性能**: 您可以通过调整 LoraConfig 和 SFTConfig 中的超参数来进一步优化模型的性能。
