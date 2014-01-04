import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool# for the python type


@cython.boundscheck(False)
@cython.wraparound(False)
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
cdef void perplexity_likelihood_function(double hyper_parameter, double tmp,
                                         int cols, double *perp_like):
    """ Perplexity and likelihood  """
    cdef unsigned int i
    for i in xrange(cols):
        perp_like[i] = float(i)
    # likelihood beta,  p(w | z, beta)  integrating out phi
    # likelihood alpha, p(z | alpha)    integrating out theta
    # likelihood alpha, beta
    # perplexity

# define a function pointer to the CGS LDA
ctypedef void (*lda_cgs_initial_ptr)(bool,
                                     unsigned int, unsigned int, unsigned int,
                                     unsigned long*, unsigned long*,
                                     unsigned int*,
                                     double*, double*, double*,
                                     void (*func_ptr)(double, double, int, double*),
                                     double*, int)

ctypedef void (*lda_cgs_ptr)(unsigned int, unsigned int, unsigned int,
                             unsigned long*, unsigned long*,
                             unsigned int*,
                             double*, double*, double*,
                             void (*func_ptr)(double, double, int, double*),
                             double*, int)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void lda_cgs_initial(bool smartinit,
                          unsigned int M, unsigned int K, unsigned int V,
                          unsigned long* docs, unsigned long* docs_start_index,
                          unsigned int* z_m_n,
                          double* n_m_z, double* n_z_t, double* n_z,
                          void (*func_ptr)(double, double, int, double*),
                          double* perp_likelihood, int n_dim):

    cdef unsigned int j
    cdef unsigned int m, t, index
    cdef unsigned int start_index, end_index
    cdef double [::1] p_z
    ascontig = np.ascontiguousarray

    for m in xrange(M):#document
        start_index = docs_start_index[m]
        end_index = docs_start_index[m+1]
        #print '%d document is (%u,%u)~(%u,%u)' % (m,                              
        #                                          start_index, docs[start_index],
        #                                          end_index-1, docs[end_index-1])
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

    # perplexity and likelihood
    func_ptr(0.1, 0.1,
             n_dim,
             perp_likelihood)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void lda_cgs(unsigned int M, unsigned int K, unsigned int V,
                  unsigned long* docs, unsigned long* docs_start_index,
                  unsigned int* z_m_n,
                  double* n_m_z, double* n_z_t, double* n_z,
                  void (*func_ptr)(double, double, int, double*),
                  double* perp_likelihood, int n_dim):

    cdef unsigned int j
    cdef unsigned int m, t, index, z, new_z
    cdef unsigned int start_index, end_index
    cdef double [::1] p_z
    ascontig = np.ascontiguousarray
    
    for m in xrange(M):
        start_index = docs_start_index[m]
        end_index = docs_start_index[m+1]
        for index in xrange(start_index, end_index):
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
             n_dim,
             perp_likelihood)


# --------------------------------------------------------------------------------------


@cython.boundscheck(False)
@cython.wraparound(False)
def main():
    
    """ 1. get_corpus """
    data = get_corpus()

    """ 2. lda inference """
    cdef unsigned int i, j, k
    # 2.1 initialize
    cdef int K,V,interations
    cdef double alpha, beta
    cdef bool smartinit
    cdef unsigned long[::1] docs, docs_start_index
    docs = np.zeros(len(data['docs']), dtype=np.uint64)
    docs_start_index = np.zeros(len(data['index']), dtype=np.uint64)

    K = data['options'].K
    V = data['voca_size']
    iterations = data['options'].iteration
    alpha = data['options'].alpha
    beta = data['options'].beta
    smartinit = data['options'].smartinit
    docs = data['docs']
    docs_start_index = np.array([ np.array(data['index'])[:j].sum() for j in xrange(1,len(data['index'])+1) ])
    del data
    
    # 2.2 inference        
    cdef unsigned int len_docs=docs.__len__(), len_docs_start_index=docs_start_index.__len__()
    cdef double pre_perp, prep
    ascontig = np.ascontiguousarray

    cdef unsigned int M=len_docs_start_index-1, N=0
    cdef unsigned int[::1] z_m_n
    cdef double[:, ::1] n_m_z, n_z_t
    cdef double[::1] n_z
    cdef double[:, ::1] perp_likelihood
        
    # Step1.1 : initialize, n_z, n_z_t, n_m_z, z_m_n
    n_m_z = np.zeros((M,K), dtype=np.double) + alpha
    n_z_t = np.zeros((K,V), dtype=np.double) + beta
    n_z = np.zeros(K, dtype=np.double) + float(V)*beta
    z_m_n = np.zeros(docs_start_index[M], dtype=np.uint32)
    perp_likelihood = np.zeros((iterations, 4), dtype=np.double)
    # データ配列にポインタでアクセス
    cdef unsigned long *docs_ptr = &docs[0]
    cdef unsigned long *docs_start_index_ptr = &docs_start_index[0]
    cdef unsigned int *z_m_n_ptr = &z_m_n[0]
    cdef double *n_m_z_ptr = &n_m_z[0,0]
    cdef double *n_z_t_ptr = &n_z_t[0,0]
    cdef double *n_z_ptr = &n_z[0]
    cdef double *perp_likelihood_ptr = &perp_likelihood[0,0]

    cdef unsigned int n_dim = perp_likelihood.shape[1]

    # Step1.2 : 初期化, 各文書の各単語にトピックを与える
    cdef lda_cgs_initial_ptr initial_cgs
    initial_cgs = &lda_cgs_initial
    lda_cgs_initial(smartinit,
                    M, K, V,
                    docs_ptr, docs_start_index_ptr,
                    z_m_n_ptr,
                    n_m_z_ptr, n_z_t_ptr, n_z_ptr,
                    perplexity_likelihood_function,
                    perp_likelihood_ptr + 0*n_dim, n_dim)

    # Step2 : CGS inference    
    cdef unsigned int iteration
    cdef lda_cgs_ptr cgs
    cgs = &lda_cgs
    for iteration in xrange(iterations):
        cgs(M, K, V,
            docs_ptr, docs_start_index_ptr,
            z_m_n_ptr,
            n_m_z_ptr, n_z_t_ptr, n_z_ptr,
            perplexity_likelihood_function,
            perp_likelihood_ptr + iteration*n_dim, n_dim)

    return None
