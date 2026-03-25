#### 2026/02/27

1. 比较的时候模型要完全一致（包括optimizer）等各种，而我们之前模型并不一致：因此，要做相同模型的不同方法的比较，可以现在synthetic dataset上看看跑的效果怎么样
2. 再找一下其它的feature selection用MLP来做的Embedded Method
3. 看看用Transformer（embedding space）来做feature selection问题的baseline有没有代码可以跑
4. 想一下gating有几种方法 - 比如把vector换成矩阵呢？