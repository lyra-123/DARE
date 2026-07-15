import pandas as pd
import math

# time_file = '/home/dell/lyra/CASVA/coding_time/coding_time_DETRAC.csv'
# RTT = 0.08  # 80ms
#
# def wait_t(buffer):
#     if buffer:
#         t = 0
#         for item in buffer:
#             t += item[1] * INFER_T
#         return t
#     else:
#         return 0
#
#
# class ExpertDP:
#     def __init__(self):
#         self.encode_time = pd.read_csv(time_file)['mean_time'].tolist()
#
#     def solve(self, net_env, chunk_config, video_encoding_time):
#         cfg_id = 0
#         video_chunk_size, video_chunk_acc, video_chunk_bit = net_env.seq_chunk_data[(net_env.SEQ_ID, net_env.video_chunk_counter, cfg_id)]
#         for cfg_id in chunk_config:
#             # qp, s, r = config
#             # cfg_id = qp * 20 + s * 5 + r
#             et = video_encoding_time[cfg_id]
#             video_chunk_size, video_chunk_acc, video_chunk_bit = net_env.seq_chunk_data[
#                 (net_env.SEQ_ID, net_env.video_chunk_counter, cfg_id)]
#             lag = self.cal_latency(net_env, video_chunk_size, et)
#             if lag <= 2:
#                 break
#         f = video_chunk_acc
#         return cfg_id, f
#
#     def cal_latency(self, net_env, video_chunk_size, encode_t):
#         # encode_t = self.encode_time[qp * 16 + s * 4 + r]
#         # encode_t = self.encode_time[qp * 20 + s * 5 + r]
#         # video_chunk_size = net_env.df.loc[(net_env.SEQ_ID, net_env.video_chunk_counter, qp, s, r), 'Size']
#
#         end = net_env.start + encode_t
#
#         # 模拟传输，cooked_bw是以秒为单位记录的
#         while True:
#             if math.ceil(end) == end:
#                 if end + 1 >= len(net_env.cooked_bw):
#                     real_bw = net_env.cooked_bw[int(end + 1) % len(net_env.cooked_bw)]
#                 else:
#                     real_bw = net_env.cooked_bw[int(end + 1)]
#                 duration = 1
#
#             else:
#                 if math.ceil(end) >= len(net_env.cooked_bw):
#                     real_bw = net_env.cooked_bw[math.ceil(end) % len(net_env.cooked_bw)]
#                 else:
#                     real_bw = net_env.cooked_bw[math.ceil(end)]
#                 duration = math.ceil(end) - end
#
#             if video_chunk_size - real_bw * 1000 * duration >= 0:
#                 video_chunk_size = video_chunk_size - real_bw * 1000 * duration
#                 end += duration
#             else:
#                 end += video_chunk_size / (real_bw * 1000)
#                 video_chunk_size = 0
#             if video_chunk_size == 0:
#                 lag = end - encode_t - net_env.start + RTT
#                 # latency = end + RTT - net_env.video_start_shoot - Length
#                 break
#
#         # # 模拟了一个服务器buffer
#         # l_e = net_env.last_end
#         # server_buff = net_env.server_bu
#         # while server_buff:
#         #     if max(l_e, server_buff[0][0]) + server_buff[0][1] * INFER_T <= end:
#         #         l_e = max(l_e, server_buff[0][0]) + server_buff[0][1] * INFER_T
#         #         server_buff.pop(0)
#         #     else:
#         #         break
#         #
#         # # 这里的latency计算的是总延迟，即从拍摄完成到服务器分析结束
#         # # 在线过程中，输入的upload delay不包含服务器处理时间
#         # latency += wait_t(server_buff) + FRAME[s] * INFER_T
#
#         return lag

class ExpertDP:
    def __init__(self):
        self.encode_time = pd.read_csv(time_file)['mean_time'].tolist()

    def solve(self, net_env, chunk_config, video_encoding_time):
        cfg_id = 0
        video_chunk_size, video_chunk_acc, video_chunk_bit = net_env.seq_chunk_data[(net_env.SEQ_ID, net_env.video_chunk_counter, cfg_id)]
        for cfg_id in chunk_config:
            # qp, s, r = config
            # cfg_id = qp * 20 + s * 5 + r
            et = video_encoding_time[cfg_id]
            video_chunk_size, video_chunk_acc, video_chunk_bit = net_env.seq_chunk_data[
                (net_env.SEQ_ID, net_env.video_chunk_counter, cfg_id)]
            lag = self.cal_latency(net_env, video_chunk_size, et)
            if lag <= 1.5:
                break
        f = video_chunk_acc
        return cfg_id, f

    def cal_latency(self, net_env, video_chunk_size, encode_t):
        # encode_t = self.encode_time[qp * 16 + s * 4 + r]
        # encode_t = self.encode_time[qp * 20 + s * 5 + r]
        # video_chunk_size = net_env.df.loc[(net_env.SEQ_ID, net_env.video_chunk_counter, qp, s, r), 'Size']

        end = net_env.start + encode_t

        # 模拟传输，cooked_bw是以秒为单位记录的
        while True:
            if math.ceil(end) == end:
                if end + 1 >= len(net_env.cooked_bw):
                    real_bw = net_env.cooked_bw[int(end + 1) % len(net_env.cooked_bw)]
                else:
                    real_bw = net_env.cooked_bw[int(end + 1)]
                duration = 1

            else:
                if math.ceil(end) >= len(net_env.cooked_bw):
                    real_bw = net_env.cooked_bw[math.ceil(end) % len(net_env.cooked_bw)]
                else:
                    real_bw = net_env.cooked_bw[math.ceil(end)]
                duration = math.ceil(end) - end

            if video_chunk_size - real_bw * 1000 * duration >= 0:
                video_chunk_size = video_chunk_size - real_bw * 1000 * duration
                end += duration
            else:
                end += video_chunk_size / (real_bw * 1000)
                video_chunk_size = 0
            if video_chunk_size == 0:
                lag = end - encode_t - net_env.start + RTT
                # latency = end + RTT - net_env.video_start_shoot - Length
                break

        # # 模拟了一个服务器buffer
        # l_e = net_env.last_end
        # server_buff = net_env.server_bu
        # while server_buff:
        #     if max(l_e, server_buff[0][0]) + server_buff[0][1] * INFER_T <= end:
        #         l_e = max(l_e, server_buff[0][0]) + server_buff[0][1] * INFER_T
        #         server_buff.pop(0)
        #     else:
        #         break
        #
        # # 这里的latency计算的是总延迟，即从拍摄完成到服务器分析结束
        # # 在线过程中，输入的upload delay不包含服务器处理时间
        # latency += wait_t(server_buff) + FRAME[s] * INFER_T

        return lag

