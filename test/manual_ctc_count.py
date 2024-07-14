'''
使用动态规划计算 CTC 损失函数的路径数

对于TOO,使用9个时间步，我们可以得到如下的对齐路径的方法：
T-OO-OOO
T-O-OOOO
T-OOO-OO
...
其中"-"表示空白字符，验证的时候，首先将空白字符之间的字符合并，然后去掉空白字符，得到最终的输出序列TOO

我们可以使用动态规划来计算这个问题。我们定义一个二维数组 dp，其中 dp[i][j] 表示在第 i 个时间步对齐到序列的第 j 个字符的路径数。
我们可以按照以下的方式填充 dp 数组：
1. 初始化 dp[0][0] = 1，表示在初始状态下，没有字符被对齐。
2. 对于每个时间步 i 和每个字符 j，我们有两种选择：
    a. 不对齐当前字符：dp[i][j] += dp[i-1][j]
    b. 对齐当前字符：dp[i][j] += dp[i-1][j-1]
3. 最终的结果是 dp[T][L]，即在 T 个时间步中对齐整个序列的路径数。

'''

def count_ctc_paths(sequence, time_steps):
    seq_len = len(sequence)
    dp = [[0] * (seq_len + 1) for _ in range(time_steps + 1)]

    # Initial state
    dp[0][0] = 1

    # Fill the dp table
    for i in range(1, time_steps + 1):
        for j in range(seq_len + 1):
            dp[i][j] += dp[i - 1][j]  # Option to not align to current character
            if j > 0:
                dp[i][j] += dp[i - 1][j - 1]  # Option to align to current character
    for row in dp:
        for col in row:
            print(col, end=" ")
        print()

    # The result is the number of ways to align the entire sequence
    return dp[time_steps][seq_len]


sequence = "TOO"
time_steps = 9
num_paths = count_ctc_paths(sequence, time_steps)
print(f"Number of alignment paths: {num_paths}")


