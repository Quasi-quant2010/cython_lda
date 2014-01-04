#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool# for the python type

# get_courpus
cdef get_corpus():
    """ 
    A python warpper for getiing coupus data 
    [input]
    iteration, K, smartinit, stopwrod, filename, beta, sed, alpha, courpus

    [output]    
    docs : np.ndarray : int
    index : np.ndarray : 各文書の長さ
    options : dictionary : {'df': 0, 'iteration': 10, 'K': 10, 'smartinit': True, 'stopwords': False, 'filename': None, 'beta': 0.5, 'seed': None, 'alpha': 0.5, 'corpus': '0:3'}
    voca_size : int
    """

    import optparse
    import vocabulary
    # 設定
    parser = optparse.OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("-c", dest="corpus", help="using range of Brown corpus' files(start:end)")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.5)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.5)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
    parser.add_option("-s", dest="smartinit", action="store_true", help="smart initialize of parameters", default=False)
    parser.add_option("--stopwords", dest="stopwords", help="exclude stop words", action="store_true", default=False)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    parser.add_option("--df", dest="df", type="int", help="threshold of document freaquency to cut words", default=0)
    (options, args) = parser.parse_args()
    if not (options.filename or options.corpus): parser.error("need corpus filename(-f) or corpus range(-c)")
    
    # コーパスダウンロード
    if options.filename:
        corpus = vocabulary.load_file(options.filename)
    else:
        corpus = vocabulary.load_corpus(options.corpus)
        if not corpus: parser.error("corpus range(-c) forms 'start:end'")
    if options.seed != None:
        np.random.seed(options.seed)

    # コーパスの単語をIDへ変換
    voca = vocabulary.Vocabulary(options.stopwords)

    cdef int j,k
    cdef int matrix_size
    matrix_size = 0    
    for doc in corpus:
        matrix_size += len(voca.doc_to_ids(doc))
    cdef np.ndarray[np.int_t, ndim=1] doc_storage = np.zeros(matrix_size, np.int)
    cdef np.ndarray[np.int_t, ndim=1] doc_index = np.zeros(len(corpus)+1, np.int)

    j = 0
    for doc in corpus:
        for element in voca.doc_to_ids(doc):
            doc_storage[j] = element
            j += 1
    j = 0
    for j in xrange(len(corpus)+1):
        doc_index[j] = 0

    j = 0
    for j in xrange(len(corpus)):
        doc_index[j+1] = len(voca.doc_to_ids(corpus[j]))

    print "corpus=%d, words=%d, K=%d, a=%f, b=%f" % (len(corpus), len(voca.vocas), options.K, options.alpha, options.beta)
    
    return {'docs':doc_storage, 'index':doc_index, 'options':options, 'voca_size':voca.size()}




cdef class LDA:

    cdef:
        int K, V, N, M
        int len_docs_start_index
        double alpha, beta
        bool smartinit
        long [::1] docs, docs_start_index, z_m_n
        double [:, ::1] n_m_z, n_z_t
        double [::1] n_z

    def __init__(self, 
                 double alpha, double beta, int K, int V,
                 long[::1] docs, int len_docs,
                 long[::1] docs_start_index, int len_docs_start_index,
                 bool smartinit):
        """
        [Index]
        K : the number of topic
        V : the number of vocabulary
        M : the number of document in corpus
        N_m : the number of word in the document m
    
        [hyperparameter]
        alpha : document-topic mixing rate
        beta : topic-word

        [storage]
        z_n : vector(1*N_m) : 1文書内にある単語に対応するトピック. 例) word=(oguri,shadai), z_n=(1,0), 1:馬, 0:馬主
        z_m_n : vectro(\sum_{m=1}^{N_m}) : topic in the condition that document is m and word is n
        n_m_z : matrix(M*K) : C_{k,m,*}, 文書別トピックカウントintegral 単語
        n_z_t : matrix(K*V) : C_{k,*,v}, トピック別単語カウントintegral 文書
        n_z : vector(1*K):C_{k,*,*}, トピックカウントintegral 文書・単語

        [Representation]
        (k,m,t) = (topic,document,word)
        C_{k,m,t} = \sum_{m=1}^{N_m} 1_{z_{m,n}=k, y_{m,n}=t}, 1-of-K表現    
        C_{k,*,t} = \sum_{m=1}^{M} C_{k,m,t} : Over the document, Indicator{トピック=k and 単語=t}, topick別単語分布計算時に使用
        C_{k,m,*} = \sum_{j=1}^{V} C_{k,m,t} : Over the topic in the document, Indicator{単語=t}, topic mixing rate計算時に使用
        C_{k,*,*} = \sum_{m=1}^{M} \sum_{t=1}^{V} C_{k,m,t} : Over the document and topic in each document, Indicator{単語=t}
        """    
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.V = V
        self.N = 0
        self.M = len_docs_start_index - 1
        self.len_docs_start_index = len_docs_start_index

        self.docs = docs
        self.docs_start_index = docs_start_index

        self.z_m_n = np.zeros(docs_start_index[len_docs_start_index-1], dtype=np.intp)

        self.n_m_z = np.zeros((self.M, K), dtype=np.double) + alpha
        self.n_z_t = np.zeros((K,V), dtype=np.double) + beta
        self.n_z = np.zeros(K, dtype=np.double) + V*beta

        # 初期化:各文書の各単語にトピックを与える
        cdef unsigned int j
        cdef unsigned int m, t, index, z
        cdef unsigned int start_index, end_index
        #cdef long [::1] z_n
        cdef double [::1] p_z
        ascontig = np.ascontiguousarray

        for m in xrange(self.M):#document
            start_index = docs_start_index[m]
            end_index = docs_start_index[m+1]
            #z_n = np.zeros((end_index - start_index), dtype=np.intp)
            #print 'document is %d~%d' % (self.docs[start_index], self.docs[end_index-1])

            for index in xrange(start_index, end_index):#word_index
                t = self.docs[index]# t is word
                p_z = np.zeros(K, np.double)# initialize p_n

                if smartinit:
                    for j in xrange(K):
                        p_z[j] = (self.n_m_z[m,j]*self.n_z_t[j,t]) / self.n_z[j]
                    z = np.random.multinomial(1,
                                              ascontig(p_z,dtype=np.double) / ascontig(p_z, dtype=np.double).sum()).argmax()
                else:
                    z = np.random.randint(0,K)
                #z_n[index] = z
                self.n_m_z[m,z] += 1
                self.n_z_t[z,t] += 1
                self.n_z[z] += 1
                #self.z_m_n[index] = z_n[index]
                self.z_m_n[index] = z

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def inference(self):
        """learning once iteration"""
        cdef unsigned int j
        cdef unsigned int m, t, index, z, new_z
        cdef unsigned int start_index, end_index
        cdef double [::1] p_z
        ascontig = np.ascontiguousarray
        
        for m in xrange(self.M):#document
            start_index = self.docs_start_index[m]
            end_index = self.docs_start_index[m+1]
            #print 'document is %d~%d' % (self.docs[start_index], self.docs[end_index-1])

            for index in xrange(start_index, end_index):#word_index
                t = self.docs[index]# t is word
                z = self.z_m_n[index]# z is topic for the word t
                
                # discount for word t with topic z in document m
                self.n_m_z[m,z] -= 1
                self.n_z_t[z,t] -= 1
                self.n_z[z] -= 1
                
                # sampling topic new_z for t
                p_z = np.zeros(self.K, np.double)
                for j in xrange(self.K):
                    p_z[j] = (self.n_m_z[m,j]*self.n_z_t[j,t]) / self.n_z[j]
                new_z = np.random.multinomial(1,
                                              ascontig(p_z,dtype=np.double) / ascontig(p_z, dtype=np.double).sum()).argmax()

                # update z by the new_z topic
                self.z_m_n[index] = new_z
                # increment counters
                self.n_m_z[m,new_z] += 1
                self.n_z_t[new_z,t] += 1
                self.n_z[new_z] += 1

# lda_learning
@cython.boundscheck(False)
@cython.wraparound(False)
cdef lda_learning(int K, int V, int iterations,
                  double alpha, double beta,
                  long[::1] docs, long[::1] docs_start_index,
                  bool smartinit):
    """
    cgs inference wrapper for LDA class
    """
    cdef int i, j, k, iteration
    cdef int len_docs, len_docs_start_index
    cdef double pre_perp, prep
    cdef LDA lda

    len_docs = docs.__len__()
    len_docs_start_index = docs_start_index.__len__()

    #ascontig = np.ascontiguousarray
    #print type( ascontig(docs_start_index,dtype=np.intp) ), ascontig(docs_start_index,dtype=np.intp)
    #print len_docs, len_docs_start_index
    
    # Step1 Call LDA and initialize
    lda = LDA(alpha, beta, K, V,
              docs, len_docs,
              docs_start_index, len_docs_start_index,
              smartinit)    
    # Step2 Learning the hyper paramter alpha, beta
    for iteration in xrange(iterations):
        # 1. cgs
        lda.inference()
    #    # 2. perplexity, likelihood
    #    #print 'iteration=%d' % iteration

    return None

def main():
    """
    A warpper for python
    """
    
    """ 1. get_corpus """
    data = get_corpus()

    """ 2. lda inference """
    # 2.1 initialize
    cdef int K,V,interation
    cdef double alpha, beta
    cdef bool smartinit
    cdef np.ndarray[np.int_t, ndim=1] docs, docs_start_index
    docs = np.zeros(len(data['docs']), np.intp)
    docs_start_index = np.zeros(len(data['index']), np.intp)

    K = data['options'].K
    V = data['voca_size']
    iteration = data['options'].iteration
    alpha = data['options'].alpha
    beta = data['options'].beta
    smartinit = data['options'].smartinit
    docs = data['docs']
    docs_start_index = np.array([ np.array(data['index'])[:j].sum() 
                                  for j in xrange(1,len(data['index'])+1) ])
    del data    

    # 2.2 inference
    lda_learning(K, V, iteration,
                 alpha, beta,
                 docs, docs_start_index,
                 smartinit)

    return None
