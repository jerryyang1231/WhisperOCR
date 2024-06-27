from zhconv import convert

# 简体中文的 predicted_words 列表
predicted_words = [['锁便听听人闲起了歌'], ['侠天的风']]

# 将 predicted_words 中的每个句子转换成繁体中文
converted_predicted_words = [[convert(sentence, 'zh-tw') for sentence in sublist] for sublist in predicted_words]

print(converted_predicted_words)

