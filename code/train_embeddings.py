org_data_path = '/iesl/canvas/aranjan/rowless/all_data/kb_eps_only'
data_path = '/iesl/canvas/aranjan/rowless/'

embeddings_size = 25

def make_text_transcript(sents_files, max_sentence_size=0, text_transcript_overwrite=True):
    if text_transcript_overwrite:
        with open(data_path+'temp/text_transcript.txt','wt') as f:
            f.write('')
    for sent_file in sents_files:
        try:
            with open(sent_file,'rb') as f:
                test_lines = [(str(codecs.unicode_escape_decode(str(i)[2:-3])[0])[1:].split('\t')) for i in f.readlines()]
            test_lines = np.array([i[-1] for i in test_lines if len(i)==13])
            max_sentence_size = max([len(i.split(' ')) for i in test_lines]+[max_sentence_size])
            text = ' '.join(test_lines)
            with open(data_path+'temp/text_transcript.txt','at') as f:
                f.write(text+' ')
        except FileNotFoundError:
            print(sent_file,'does not exist')
        gc.collect()
    return max_sentence_size

files = [(org_data_path+'kb_0'+str(i)) if i<10 else (org_data_path+'kb_'+str(i)) for i in range(20)]
max_sent_size = make_text_transcript(files)

import fasttext
embeddings_model = fasttext.skipgram(data_path+'temp/text_transcript.txt',data_path+'temp/embeddings_model_2',min_count=1,dim=embeddings_size)