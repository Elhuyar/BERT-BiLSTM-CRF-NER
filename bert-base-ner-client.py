import time
from bert_as_server.client import BertClient


def bert2conll(tokens, labels):
    tok = [item for sublist in tokens for item in sublist]
    lab = [item for sublist in labels for item in sublist]

    print (tok)
    print (lab)
    print("number of tokens in input({}) and predicted labels ({})".format(len(tok),len(lab)))
    if len(tok)!=len(lab):
        print("number of tokens in input({}) and predicted labels are different({})".format(len(tok),len(lab)))
        return None
    else:
        toprint_tok=""
        toprint_tag=""
        result=[]        
        for t,l in zip(tok,lab):
            wrdpiece=t.decode("utf8")
            tag=l
            #print("{} {}\n".format(wrdpiece,tag))
            
            if tag == "X":
                toprint_tok+=wrdpiece.lstrip('#')
            else:
                if i > 0:
                    result.append("{} {}".format(toprint_tok,toprint_tag))

                if wrdpiece == "[CLS]":
                   toprint_tok=""
                   toprint_tag=""                   
                else:
                    toprint_tok=wrdpiece
                    toprint_tag=tag
                
        return result

with BertClient(show_server_config=True, check_version=False, check_length=False, mode='NER') as bc:
    start_t = time.perf_counter()
    str = 'Gaur Donostian Errealaren partidua dago, Osasunaren aurka, arratsaldeko 19:00tan. Iñigo Urkullu ere joatekoa da.'
    rst = bc.encode([str])
    #print(rst)
    for r in range(len(rst)):
        tok=rst[r]['tokens']
        lab=rst[r]['pred_label']

        print('\n'.join(bert2conll(tok,lab)),"\n")
    #print('rst:', rst)
    print(time.perf_counter() - start_t)
