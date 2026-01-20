#<h1 align="center">强化学习笔记</h1>
#<h1 align="left">M1基础</h1>
#1.强化学习与有监督学习的区别，两者目标不同
#<img width="1172" height="281" alt="image" src="https://github.com/user-attachments/assets/3b5fad11-bfeb-43ca-8504-5822c72da814" />
<img width="821" height="250" alt="image" src="https://github.com/user-attachments/assets/618a2363-5e1c-4af2-9647-54e785dc423d" />
<img width="830" height="327" alt="image" src="https://github.com/user-attachments/assets/55626caf-0830-4c2c-9273-36c71f7b5941" />

#2.强化学习中的占用度量
</br>
<img width="568" height="307" alt="image" src="https://github.com/user-attachments/assets/648b438e-c655-4e6f-b81d-eb873b3ab9d4" />
</br>
#3.强化学习中环境的下一刻状态的概率分布将由当前状态和智能体的动作来共同决定，用最简单的数学公式表示则是
</br>
<img width="391" height="50" alt="image" src="https://github.com/user-attachments/assets/b90e8286-8386-4c5f-a05d-669c140924b8" />
#<h1 align="left">M2多臂老虎机</h1>
多臂老虎机可以看作一种简化版的强化学习，他不存在状态信息，只有动作和奖励，学习这个有利于理解强化学习
#<h2 align="left">M2.1问题介绍</h2>
1.问题定义，K根拉杆的老虎机，每一个杆子都对应一个未知的奖励分布，从头尝试，目的是T次拉杆之后获得最多的奖励。奖励分布是未知的，所以需要衡量“探索概率”还是“根据经验选择获奖次数多的”
<img width="587" height="607" alt="image" src="https://github.com/user-attachments/assets/cf0ffc90-fe23-4dea-b34b-460476092d5a" />
</br>
2.形式化描述
</br>
<img width="592" height="322" alt="image" src="https://github.com/user-attachments/assets/9aae82a5-38ef-43d3-aa9a-5dfc7e55610d" />
<img width="382" height="47" alt="image" src="https://github.com/user-attachments/assets/e7e540c7-d42c-4205-b369-5b86bf5f8464" />
</br>
<img width="671" height="267" alt="image" src="https://github.com/user-attachments/assets/1aad1398-695e-46c9-b824-26b4750f0596" />
<img width="808" height="556" alt="image" src="https://github.com/user-attachments/assets/0c8f6928-e51b-4bdc-a5e4-0997e478bbee" />
</br>
3.累积懊悔
</br>
<img width="563" height="316" alt="image" src="https://github.com/user-attachments/assets/004396be-41a4-4d78-b50b-037c0d7b5426" />
</br>
########
4.重点： 估计期望奖励
</br>
<img width="593" height="175" alt="image" src="https://github.com/user-attachments/assets/5b8b5f9f-8d58-4c71-ad4d-1245d4ccaf8a" />
</br>
这里可以理解为<img width="47" height="37" alt="image" src="https://github.com/user-attachments/assets/85fdf20e-061c-498b-a4f7-3b7566ffa371" />为平均奖励，k轮后的平均奖励，这里可理解为前k-1轮的奖励再加上k轮单独的奖励除以k轮，将每一轮的奖励平均到每一轮
</br>
编写代码来实现一个拉杆数为 10 的多臂老虎机。其中拉动每根拉杆的奖励服从伯努利分布（Bernoulli distribution），即每次拉下拉杆有p的概率获得的奖励为 1，有的1-p概率获得的奖励为 0。奖励为 1 代表获奖，奖励为 0 代表没有获奖。
</br>
<img width="973" height="777" alt="image" src="https://github.com/user-attachments/assets/3904f9fc-98fc-40d4-bcea-cd5697eae45b" />
</br>
代码解释1：self.best_idx = np.argmax(self.probs) argmax返回的是编号
</br>
代码解释2：self.best_prob = self.probs[self.best_idx]  通过索引来获得最大概率
</br>
代码解释3：if np.random.rand() < self.probs[k]: return 1  功能：生成一个 [0,1) 的随机浮点数用来模拟一次随机试验
概率大于随机数即中奖
</br>






