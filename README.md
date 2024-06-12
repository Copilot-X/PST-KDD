# 1、基本环境
    Python版本： Python 3.10.12
    cuda版本： 12.2
    cudnn： 8.3.1
    显卡： RTX-3090-24G
    安装依赖包：
        pip install -r requirements.txt

# 2、下载预训练模型，放置在pretrain_models目录下
    scideberta-cs 下载链接:https://huggingface.co/KISTI-AI/scideberta-cs/tree/main

# 3、下载微调好的预训练模型，解压放置在outputs目录下
    链接：https://pan.baidu.com/s/1mPXoc_SrfnVvfmeVFrEDtA?pwd=skw7  提取码：skw7

# 4、数据处理 & 训练脚本
    ① 下载官方数据集paper-xml放置data目录下
    ② sh train.sh

# 5、推理脚本
    sh predict.sh
    生成结果在当前目录 result.json

# 6、若过程有疑问/问题，请联系：1506025911@qq.com