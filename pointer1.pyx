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
    cdef np.ndarray[np.uint64_t, ndim=1] doc_storage = np.zeros(matrix_size, np.uint64)
    cdef np.ndarray[np.uint64_t, ndim=1] doc_index = np.zeros(len(corpus)+1, np.uint64)

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

#--------------------------------- LDA --------------------------------------

cdef extern from 'stdlib.h':
    ctypedef unsigned long size_t
    void *malloc(size_t size)
    void free(void *prt)

#cdef extern from 'stdbool.h':
#    ctypedef bool boolean

cdef extern from "gsl/gsl_statistics_double.h":
    size_t gsl_stats_max_index(double data[],
                               size_t stride,
                               size_t n)

cdef extern from "gsl/gsl_rng.h":
    ctypedef struct gsl_rng_type:
        pass
    ctypedef struct gsl_rng:
        pass
    gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T)

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

cdef extern from "gsl/gsl_randist.h":
    void multinomial "gsl_ran_multinomial"(gsl_rng * r,
                                           size_t cluster,
                                           unsigned int N,
                                           double *theta,
                                           unsigned int *n)
    void dirichlet "gsl_ran_dirichlet"(gsl_rng * r,
                                       size_t cluster,
                                       double *alpha,
                                       double *theta)

# define a function pointer to the CGS LDA
ctypedef void (*lda_ptr)(double, double, int, int, int,
                         unsigned long*, int,
                         unsigned long*, int,
                         bool,
                         void (*func_ptr)(double, double, int, double*),
                         double)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void Topic_Word_Dist(double *n_z_t, int K, int V,
                          double *n_z, 
                          double *p_n_z_t):
    """ Topic-Wrod distribution  """
    cdef unsigned int i,j
    for i in xrange(K):
        for j in xrange(V):
            p_n_z_t[i*V + j] = n_z_t[i*V + j] / n_z[i]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void Document_Topic_Dist(double *n_m_z, int K, int M,
                              double *n_z, 
                              double *p_n_m_z):
    """ Topic-Wrod distribution  """
    cdef unsigned int i,j
    for i in xrange(M):
        for j in xrange(K):
            p_n_m_z[i*K + j] = n_m_z[i*K + j] / n_z[i]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void perplexity_likelihood(double hyper_parameter, double tmp,
                                int cols, double *perp_like):
    cdef unsigned int i
    for i in xrange(cols):
        perp_like[i] = float(i)
    # likelihood beta,  p(w | z, beta)  integrating out phi
    # likelihood alpha, p(z | alpha)    integrating out theta
    # likelihood alpha, beta
    # perplexity

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void lda(double alpha, double beta, int K, int V, int iterations,
              unsigned long *docs, int len_docs,
              unsigned long *docs_start_index, int len_docs_start_index,
              bool smartinit,
              void (*func_ptr)(double, double, int, double*),
              double zz):
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
    z_n : vector(1*N_m) : 1文書内にある単語に対応するトピック
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
    cdef unsigned int i, j, k, iteration
    cdef unsigned int M=len_docs_start_index-1, N=0
    cdef unsigned int *z_m_n
    cdef double *n_m_z, *n_z_t, *n_z
    cdef double[:,::1] perp_likelihood = np.zeros((iterations, 4), dtype=np.double)
    ascontig = np.ascontiguousarray

    # Step1.1 : initialize, n_z, n_z_t, n_m_z, z_m_n
    n_m_z = <double*>malloc((K*M) * sizeof(double))
    n_z_t = <double*>malloc((K*V) * sizeof(double))
    n_z = <double*>malloc(K * sizeof(double))
    z_m_n = <unsigned int*>malloc(docs_start_index[M] * sizeof(unsigned int))
    if not n_m_z or not n_z_t or not n_z or not z_m_n:
        raise MemoryError("Cannot allocate memory")
    for i in xrange(docs_start_index[M]):
        z_m_n[i] = 0
    for i in xrange(K):
        n_z[i] = 0.0 + float(V) * beta
    for i in xrange(K):
        for j in xrange(V):
            n_z_t[i*V + j] = 0.0 + beta
    for i in xrange(M):
        for j in xrange(K):
            n_m_z[i*K + j] = 0.0 + alpha

    # Step1.2 : 初期化, 各文書の各単語にトピックを与える
    cdef unsigned int m, t, index, z, new_z
    cdef unsigned int start_index, end_index
    cdef double [::1] p_z    
    for m in xrange(M):#document
        start_index = docs_start_index[m]
        end_index = docs_start_index[m+1]
        """
        print '%d document is (%u,%u)~(%u,%u)' % (m,
                                                  start_index, docs[start_index],
                                                  end_index-1, docs[end_index-1])
        """
        for index in xrange(start_index, end_index):#word_index
            t = docs[index]# t is word
            p_z = np.zeros(K, np.double)# initialize p_n
            if smartinit:
                for j in xrange(K):
                    p_z[j] = n_m_z[m*K + j] * n_z_t[j*V + t] / n_z[j]
                z = np.random.multinomial(1,
                                          ascontig(p_z,dtype=np.double) / ascontig(p_z,dtype=np.double).sum()).argmax()
            else:
                z = np.random.randint(0,K)
            n_m_z[m*K + z] += 1
            n_z_t[z*V + t] += 1
            n_z[z] += 1
            z_m_n[index] = z

    # Step1.3 : initial perplexity and likelihood
    cdef double *perp_likelihood_ptr = &perp_likelihood[0, 0] #perp_likelihood[iteration=0, 0]
    cdef unsigned int n_dim = perp_likelihood.shape[1]
    func_ptr(0.1, 0.1,
             perp_likelihood.shape[1],
             perp_likelihood_ptr + 0*n_dim)    

    # Step2 : CGS inference    
    for iteration in xrange(1,iterations):
        for m in xrange(M):
            start_index = docs_start_index[m]
            end_index = docs_start_index[m+1]
            for index in xrange(start_index, end_index):#word_index
                t = docs[index]# t is word
                z = z_m_n[index]# z is topic for the word t

                # discount for word t with topic z in document m
                n_m_z[m*K + z] -= 1
                n_z_t[z*V + t] -= 1
                n_z[z] -= 1

                # sampling topic new_z for t
                p_z = np.zeros(K, np.double)
                for j in xrange(K):
                    p_z[j] = n_m_z[m*K + j] * n_z_t[j*V + t] / n_z[j]
                new_z = np.random.multinomial(1,
                                              ascontig(p_z,dtype=np.double) / ascontig(p_z,dtype=np.double).sum()).argmax()
            
                # update z by the new_z topic
                z_m_n[index] = new_z

                # increment counters
                n_m_z[m*K + new_z] += 1
                n_z_t[new_z*V + t] += 1
                n_z[new_z] += 1

        # caculate perplexity and likelihood
        func_ptr(0.1, 0.1, 
                 perp_likelihood.shape[1],
                 perp_likelihood_ptr + iteration*n_dim)

    #print ascontig(perp_likelihood, dtype=np.double)

    # Step3 : free
    if z_m_n:
        free(z_m_n); z_m_n == NULL
    if n_m_z:
        free(n_m_z); n_m_z == NULL
    if n_z_t:
        free(n_z_t); n_z_t == NULL
    if n_z:
        free(n_z); n_z == NULL

    zz =  1.0

# lda_learning
@cython.boundscheck(False)
@cython.wraparound(False)
cdef lda_learning(int K, int V, int iterations,
                  double alpha, double beta,
                  unsigned long[::1] docs, unsigned long[::1] docs_start_index,
                  bool smartinit):
    """
    cgs inference wrapper for LDA 
    """
    cdef int i, j, k, iteration
    cdef int len_docs, len_docs_start_index
    cdef double pre_perp, prep

    len_docs = docs.__len__()
    len_docs_start_index = docs_start_index.__len__()

    # LDA
    cdef lda_ptr lda_inference
    lda_inference = &lda

    # 配列にポインターを使ってアクセス
    #ZptrにDの先頭アドレスを渡す.　ただし、cdef double *Zptr = &Z はエラー
    cdef unsigned long *docs_ptr = &docs[0]
    cdef unsigned long *docs_start_index_ptr = &docs_start_index[0]

    lda_inference(alpha, beta, K, V, iterations,
                  docs_ptr, len_docs,
                  docs_start_index_ptr, len_docs_start_index,
                  smartinit,
                  perplexity_likelihood,
                  prep)

    return None


def main():
    """ A warpper for python """

    
    """ 1. get_corpus """
    data = get_corpus()

    """ 2. lda inference """
    # 2.1 initialize
    cdef int K,V,interation
    cdef double alpha, beta
    cdef bool smartinit
    cdef np.ndarray[np.uint64_t, ndim=1] docs, docs_index
    docs = np.zeros(len(data['docs']), dtype=np.uint64)
    docs_index = np.zeros(len(data['index']), dtype=np.uint64)

    K = data['options'].K
    V = data['voca_size']
    iteration = data['options'].iteration
    alpha = data['options'].alpha
    beta = data['options'].beta
    smartinit = data['options'].smartinit
    docs = data['docs']
    docs_index = np.array([ np.array(data['index'])[:j].sum() for j in xrange(1,len(data['index'])+1) ])
    del data    
    
    # 2.2 inference
    lda_learning(K, V, iteration,
                 alpha, beta,
                 docs, docs_index,
                 smartinit)
    
    return None
