'''n-gram models'''

'''Create n-grams for features extraction (unigram, bigram, trigram)'''

'''Create unigram
Eg. input list of words: ['My', 'research', 'project', 'code']
    Unigram ouput: ['My', 'research', 'project', 'code']   '''
def unigram(transcript_words):
     assert type(transcript_words) == list
     return transcript_words # Return unigram



'''Create Bigram model
Eg. input list of words: ['My', 'research', 'project', 'code']
    Bigram ouput: ['My_research', 'research_project', 'project_code']'''

# Create Bigrams
def bigram(transcript_words, string, skip_word=0):
    assert type(transcript_words) == list
    List = len(transcript_words)
    if List > 1:
        Lst = []
        for i in range(List-1):
            for x in range(1, skip_word+2):
                if i+x < List:
                    Lst.append(string.join([transcript_words[i], 
                                                 transcript_words[i+x]]))
    else:
        Lst = unigram(transcript_words) #Return unigram
    return Lst #Return Bigram

''' Create Trigrams    
Eg. input list of words: ['My', 'research', 'project', 'code']
    Trigram ouput: ['My_research_project', 'research_project_code']'''
     
def trigram(transcript_words, string, skip_word=0):
	assert type(transcript_words) == list
	length = len(transcript_words)
	if length > 2:
		Lst = []
		for i in range(length-2):
			for y1 in range(1,skip_word+2):
				for y2 in range(1,skip_word+2):
					if i+y1 < length and i+y1+y2 < length:
						Lst.append( string.join([transcript_words[i], transcript_words[i+y1], 
                                 transcript_words[i+y1+y2]]) )
	else:
		Lst = bigram(transcript_words, string, skip_word) #return Bigrams 
	return Lst # Return trigrams



