未完成  
shuyoさんのlda.py  
https://github.com/shuyo/iir/blob/master/lda/lda.py  
をCythonに移植することを目指す。  
特に、推論部分;  
For Iteration  
    For Document  
        For Word  
            # discount for word t with topic z in document m  
            # sampling topic new_z for t  
            # update z by the new_z topic  
    # caculate perplexity and likelihood  
を高速化したい  
