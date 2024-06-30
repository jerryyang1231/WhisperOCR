import re

# 定義 arabic_to_chinese 函數
def arabic_to_chinese(number):
    # 定義阿拉伯數字到中文數字的對應
    chinese_numerals = {
        '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
        '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
    }
    
    # 定義中文數字單位
    chinese_units = ['', '十', '百', '千', '萬', '十萬', '百萬', '千萬', '億']
    
    # 將整數部分轉換為中文數字
    def convert_integer_part(num_str):
        length = len(num_str)
        chinese_str = ''
        for i, digit in enumerate(num_str):
            if digit != '0':
                chinese_str += chinese_numerals[digit]
                if length - 1 - i < len(chinese_units):
                    chinese_str += chinese_units[length - 1 - i]
            elif len(chinese_str) > 0 and chinese_str[-1] != chinese_numerals['0']:
                chinese_str += chinese_numerals[digit]
        return chinese_str

    # 將小數部分轉換為中文數字
    def convert_decimal_part(num_str):
        return ''.join(chinese_numerals[digit] for digit in num_str)

    # 將阿拉伯數字轉換為中文數字的主要函數
    def convert_to_chinese(num_str):
        if '.' in num_str:
            integer_part, decimal_part = num_str.split('.')
            chinese_str = convert_integer_part(integer_part) + '點' + convert_decimal_part(decimal_part)
        else:
            chinese_str = convert_integer_part(num_str)
        
        return chinese_str

    # 判斷輸入是整數還是字符串
    if isinstance(number, int):
        return convert_to_chinese(str(number))
    elif isinstance(number, str):
        return re.sub(r'\d+(\.\d+)?', lambda x: convert_to_chinese(x.group()), number)
    return number

# # 將 arabic_to_chinese 應用到 predicted_words 中
# # predicted_words = [['2015年9月11日'], ['2015年4月20 '], ['夏威夷事件2015年10月10日'], ['2月12日晚11點過']]
# predicted_words = [['8703'], ['3、2、3、8、6、3']]
# converted_predicted_words = [[arabic_to_chinese(word) for word in sentence] for sentence in predicted_words]

# # 輸出轉換結果
# print(converted_predicted_words)

# import re

# # 定義阿拉伯數字到中文數字的對應
# chinese_numerals = {
#     '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
#     '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
# }

# # 定義 convert_date_part 函數
# def convert_date_part(date_str):
#     # 先處理年份，將四位數的年份轉換為中文數字
#     date_str = re.sub(r'(\d{4})年', lambda x: ''.join(chinese_numerals[digit] for digit in x.group(1)) + '年', date_str)
    
#     # 轉換其他數字
#     def convert_other_numbers(match):
#         num_str = match.group()
#         # 將1-9的數字轉換為中文數字
#         if len(num_str) == 1:
#             return chinese_numerals[num_str]
#         # 將10-19的數字轉換為口語化的讀法
#         elif num_str.startswith('1') and len(num_str) == 2:
#             return '十' + chinese_numerals[num_str[1]] if num_str[1] != '0' else '十'
#         # 將20-99的數字轉換為口語化的讀法
#         elif len(num_str) == 2:
#             return chinese_numerals[num_str[0]] + '十' + chinese_numerals[num_str[1]] if num_str[1] != '0' else chinese_numerals[num_str[0]] + '十'
#         return ''.join(chinese_numerals[digit] for digit in num_str)

#     # 使用正則表達式轉換日期中的其他數字
#     date_str = re.sub(r'\d+', convert_other_numbers, date_str)

#     return date_str

# # 測試例子
# print(convert_date_part('2015年9月11日'))  # 二零一五年九月十一日
# print(convert_date_part('2015年4月20日'))  # 二零一五年四月二十日
# print(convert_date_part('2月12日晚11點過'))  # 二月十二日晚十一點過

