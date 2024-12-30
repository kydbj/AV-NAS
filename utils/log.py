import json
import copy

def combine_genotype_probs(operations, probabilities):

    combined = {}

    for key in operations:
        combined[key] = {}
        for sub_key in operations[key]:
            combined[key][sub_key] = []
            for ops, probs in zip(operations[key][sub_key], probabilities[key][sub_key]):
                combined[key][sub_key].append(list(zip(ops, probs)))
    # # 将结果打印成 JSON 格式，便于阅读
    # combined_json = json.dumps(combined, indent=4)
    return combined


def log_training_data(data, file_path):
    """
    将训练数据记录到指定文件中。

    参数：
    - data: 要记录的数据，可以是字符串或其他可转换为字符串的对象。
    - file_path: 要记录数据的文件路径。
    """
    # 打开文件，使用 'a' 模式以追加的方式写入数据
    with open(file_path, 'a+') as file:
        # 将数据转换为字符串并写入文件
        file.write(str(data) + '\n')

    print(data)


if __name__ == "__main__":
    operations = {
        'ImgEnc': {
            'ModelingOperators': [['std_conv_5', 'std_conv_3'], ['std_conv_5', 'std_conv_3'],
                                  ['std_conv_5', 'std_conv_3'], ['std_conv_5', 'std_conv_3'],
                                  ['std_conv_5', 'std_conv_3']],
            'BinaryOperators': [['add'], ['add']],
            'MonadicOperators': [['skip_connect'], ['skip_connect'], ['skip_connect'], ['skip_connect'],
                                 ['skip_connect']]
        },
        'AudEnc': {
            'ModelingOperators': [['std_conv_5', 'std_conv_3'], ['std_conv_5', 'std_conv_3'],
                                  ['std_conv_5', 'std_conv_3'], ['std_conv_5', 'std_conv_3'],
                                  ['std_conv_5', 'std_conv_3']],
            'BinaryOperators': [['add'], ['add']],
            'MonadicOperators': [['skip_connect'], ['skip_connect'], ['skip_connect'], ['skip_connect'],
                                 ['skip_connect']]
        },
        'Inter': {
            'InterOperators': [['cro_att_64'], ['cro_att_64']]
        },
        'Fusion': {
            'FusionOperators': [['add']]
        }
    }

    probabilities = {
        'ImgEnc': {
            'ModelingOperators': [[0.1429, 0.1429], [0.1429, 0.1429], [0.1429, 0.1429], [0.1429, 0.1429],
                                  [0.1429, 0.1429]],
            'BinaryOperators': [[0.5], [0.5]],
            'MonadicOperators': [[0.3333], [0.3333], [0.3333], [0.3333], [0.3333]]
        },
        'AudEnc': {
            'ModelingOperators': [[0.1429, 0.1429], [0.1429, 0.1429], [0.1429, 0.1429], [0.1429, 0.1429],
                                  [0.1429, 0.1429]],
            'BinaryOperators': [[0.5], [0.5]],
            'MonadicOperators': [[0.3333], [0.3333], [0.3333], [0.3333], [0.3333]]
        },
        'Inter': {
            'InterOperators': [[0.3333], [0.3333]]
        },
        'Fusion': {
            'FusionOperators': [[1.0]]
        }
    }

    # 调用函数合并数据
    # combined_data = combine_genotype_probs(operations, probabilities)

    # 输出合并后的JSON数据
    # print(json.dumps(combined_data, indent=4))
    # print(json.dumps(genotype, indent=4))
    # print(json.dumps(probs, indent=4))


