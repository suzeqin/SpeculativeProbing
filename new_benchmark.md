| 数据集 | 来源 | 数据集属性 | 实验定位 | 可测量指标 |
| ---- | ---- | ---- | ---- | ---- |
| AgentLAB | [https://github.com/TanqiuJiang/AgentLAB](https://github.com/TanqiuJiang/AgentLAB) | 黑盒动态长程间接注入攻击数据集 | 攻击专项测试集，模拟RAG外部Token污染、网页/邮件隐性投毒等真实间接注入场景，用于考核模型抗劫持能力与探针检测能力 | 漏报率、ASR攻击成功率、误报率、投机解码接收率、接收长度、端到端时延、聚合吞吐 |
| BFCL-v4 | [https://gorilla.cs.berkeley.edu/leaderboard.html](https://gorilla.cs.berkeley.edu/leaderboard.html) | 正常工具调用数据集 | 误报率核心测试集，纯净无攻击的正常Agent工具调用样本，用于验证探针不会误伤正常业务流程 | 误报率、投机解码接收率、接收长度、端到端时延、聚合吞吐 |
| LiveCodeBench | [https://livecodebench.github.io/](https://livecodebench.github.io/) | 白盒代码生成基准数据集（参考DSpark论文） | 投机解码性能专项测试集，依托长代码生成高负载场景，校验安全探针引入后，模型投机加速性能的衰减程度 | 投机解码接收率、接收长度、端到端时延、聚合吞吐 |