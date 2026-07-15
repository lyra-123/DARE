## constants_fit.py

用Rule-Based方案求解之前，要用真实数据拟合两个模型中的参数，公式(2)、(5)

## Rule_Based.py

文中考虑的延迟是transmission latency，见公式(4)，按照作者的逻辑设计Rule_Based方案仅考虑这部分延迟。

train_trace(test_trace)和CASVA中的是一样的，这里就没有再发

## IL.py & IL_train.py

将Rule_Based方案转换成IL。输入我没有加pt（推理时间），可以补上。

## PPO.py & PPO_train.py

和CASVA一样，输入也没有加pt，同理可以补上。
原本Actor网络用一个Noise Layer代替了FC Layer，我试了一下不如原来的好，所以就没改，这里可以按照原文的描述调一调

## Fusion.py & fusion_train.py

网络结构参考Fig.7 写的。这部分按照我的理解写的，也不是很确定是不是原文的意思。
实际没有和Rule-based action distribution做融合，这里也可以按照原文改一下看看。