import rdflib as rdflib
import urllib.parse


if __name__ == '__main__':
    g = rdflib.Graph()
    g.parse('data/covid19_kb/kg.nt', format='nt')
    dois = ['10.1016/j.eng.2020.03.007', '10.3760/cma.j.cn112338-20200221-00144', '10.1128/JVI.05050-11',
            '10.1128/JVI.01570-14', '10.1016/S2214-109X(20)30065-6', '10.1101/2020.01.30.927871']
    count_dict = dict()
    cnt = 0
    for s, p, o in g:
        cnt += 1
        for doi in dois:
            s = urllib.parse.unquote(s)
            o = urllib.parse.unquote(o)
            if (doi in s or doi in o) and 'cites' in str(p).lower():
                if doi not in count_dict:
                    count_dict[doi] = 0
                count_dict[doi] += 1
        if cnt % 1000 == 0:
            print(cnt, ' Done')
    for k, v in count_dict.items():
        print(k, ' : ', v)
