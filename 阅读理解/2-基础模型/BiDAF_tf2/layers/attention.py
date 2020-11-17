import tensorflow as tf

class C2QAttention(tf.keras.layers.Layer):

    def call(self, similarity, qencode):
        # 1. 对qecncode进行扩展维度 ：tf.expand_dims
        # 2. softmax函数处理相似度矩阵：tf.keras.activations.softmax
        # 3. 对处理结果扩展维度：tf.expand_dims
        # 4. 加权求和：tf.math.reduce_sum

        return c2q_att

class Q2CAttention(tf.keras.layers.Layer):

    def call(self, similarity, cencode):

        # 1.计算similarity矩阵最大值：tf.math.reduce_max
        # 2.使用 softmax函数处理最大值的相似度矩阵：tf.keras.activations.softmax
        # 3.维度处理：tf.expand_dims
        # 4.加权求和：tf.math.reduce_sum
        # 5.再次维度处理加权求和后的结果：tf.expand_dims
        # 6.获取重复的次数： cencode.shape[1]
        # 7.重复拼接获取最终矩阵：tf.tile



        return q2c_att
