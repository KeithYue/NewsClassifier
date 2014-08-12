# coding=utf-8
# this file is used to tokenize the long documents
import jieba
jieba.load_userdict("meta/userdict.txt")
# import jieba.posseg as pseg

STOP_WORDS_PATH = './meta/stopwords.txt'

def get_stopwords():
    s = []
    f = open(STOP_WORDS_PATH, 'r')
    for k in f.readlines():
        # k = str(k, 'utf-8') # convert the source str into unicode for compare
        # print type(k)
        s.append(k.strip())

    return set(s)


def tokenize(text):
    '''
    Given a text, tokenize it into toknes, and return a list as output
    '''
    r = []
    seg_list = jieba.cut(text.strip())
    # seg_list = pseg.cut(test.strip())
    for token in seg_list:
        token = token.strip()
        if token != '' and token not in stops:
            # print type(token)
            r.append(token)
    return r


def print_tokens(tokens):
    print('/'.join(tokens))
    return

def test():
    content = ''' 目前，国际制度已经成为现代国际关系的重要特征，平安银行广泛存在于国际贸易、金融、环境、能源及国际安全等领域。国际能源制度是指在能源这一特定领域国际角色在认识上趋于一致的原则、规范、规则和决策程序。用理性主义的理论解释，国际能源制度可以帮助国家交流能源信息，减少谈判成本和交易成本，从而增加能源合作的可能性。用建构主义解释，国际能源制度可以塑造国家对能源发展的认同，并进而影响国家的行为，从而能够增进国际能源合作。国际能源合作是指能源资源国与能源消费国以及能源中转国之间进行的能源交往。加强国际合作，增加国内能源政策和市场信息的稳定性和透明度，可以消除其他国家的猜疑和担心，促进国际市场稳定。对于能源出口国而言，能源合作的目的在于保证本国的能源商品以合理而稳定的价格出售。对于中国这样的能源进口国而言，国际能源合作的目标在于在合理的价格水平下保障稳定的能源供应。
    '''
    tokens = tokenize(content)
    print('/'.join(tokens))
    return

stops = get_stopwords()

if __name__ == '__main__':
    test()
