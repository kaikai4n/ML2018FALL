import pickle
import numpy as np

class DataLoader():
    def __init__(self, 
            create_word_dict=True, 
            filenames=None, 
            word_dict_filename=None,
            save_word_dict=False):
        self.__create_word_dict = create_word_dict
        if self.__create_word_dict:
            if filenames is None:
                raise Exception('"filenames" for creating word dictionary \
                        index is not given')
            self._create_word_dict(filenames)
        else:
            if word_dict_filename is None:
                raise Exception('"word_dict_filename" for loading word \
                        dictionary index is not given')
            self._load_word_dict(word_dict_filename)
        if save_word_dict:
            if word_dict_filename is None:
                raise Exception('"word_dict_filename" saving word dict\
                        is not given')
            with open(word_dict_filename, 'wb') as f:
                pickle.dump(self._word_dict, f)


    def _load_word_dict(self, filename):
        with open(filename, 'rb') as f:
            self._word_dict = pickle.load(f)

    def _create_word_dict(self, filenames):
        self._word_dict = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<UNK>': 3}
        for filename in filenames:
            content = self.read_csv(filename)
            for ele in content:
                if ele not in self._word_dict:
                    self._word_dict[ele] = len(self._word_dict)
    
    def get_word_dict(self):
        return self._word_dict

    def read_csv(self, filename, encoding='utf-8'):
        with open(filename, 'r', encoding=encoding) as f:
            content = f.read()
        return content

    def load_data_x(self, filename, encoding='utf-8'):
        print('Loading Data...')
        content = self.read_csv(filename, encoding).split('\n')
        content = [ele.split(',', 1)[-1] for ele in content]
        del content[-1]
        del content[0]
        content = self._to_one_hot_value(content)
        # It takes most of the time to convert list to numpy
        # not worthwhile to do so since it is only necessary 
        # for one to convert to torch tensor before going 
        # through the network.
        #print('To numpy...')
        #content = self._to_numpy(content)
        return content

    def _to_one_hot_value(self, content, max_sentence_len=2000):
        print('To word dictionary value...')
        transformed_content = [[self._word_dict['<SOS>']] + \
                [self._word_dict[word] for word in line] \
                if len(line) <= max_sentence_len else \
                [self._word_dict[word] for word in line][:max_sentence_len-1]+ \
                [self._word_dict['<EOS>']] for line in content]
        print('Start padding...')
        sentence_length = [len(sentence) for sentence in transformed_content]
        max_length = max(sentence_length)
        padded_content = self._pad_equal_length(transformed_content,\
                max_length)
        return padded_content

    def _pad_equal_length(self, content, length):
        padded_content = [ele + (length-len(ele))*[self._word_dict['<PAD>']] \
                for ele in content]
        return padded_content

    def _to_numpy(self, content):
        return np.asarray(content)

if __name__ == '__main__':
    # example for loading word dictionary from file
    dl = DataLoader(
            create_word_dict=False,
            word_dict_filename='word_dict.pkl') 
    # example to create own word dictionary via data
    dl = DataLoader(
            create_word_dict=True, 
            filenames=['data/train_x.csv', 'data/test_x.csv'],
            save_word_dict=True,
            word_dict_filename='word_dict.pkl')
    train_x = dl.load_data_x('data/train_x.csv')
    #print(train_x.shape)
    print(train_x[:5][:20])
    
