# Define an array of tasks

# The task's llh score is averaged by default
export TASKS=(
#  "benczechmark_propaganda_argumentace"
#  "benczechmark_propaganda_fabulace"
#  "benczechmark_propaganda_nazor"
#  "benczechmark_propaganda_strach"
#  "benczechmark_propaganda_zamereni"
#  "benczechmark_propaganda_demonizace"
#  "benczechmark_propaganda_lokace"
#  "benczechmark_propaganda_relativizace"
#  "benczechmark_propaganda_vina"
#  "benczechmark_propaganda_zanr"
#  "benczechmark_propaganda_emoce"
#  "benczechmark_propaganda_nalepkovani"
#  "benczechmark_propaganda_rusko"
#  "benczechmark_sentiment_mall"
#  "benczechmark_sentiment_fb"
#  "benczechmark_sentiment_csfd"
#  "benczechmark_summarization"
#  "benczechmark_grammarerrorcorrection"
#  "benczechmark_cs_naturalquestions"
#  "benczechmark_cs_sqad32"
#  "benczechmark_cs_triviaQA"
#  "benczechmark_csfever_nli"
#  "benczechmark_ctkfacts_nli"
#  "benczechmark_cs_ner"
#  "benczechmark_hellaswag"
#  "benczechmark_klokan_qa"
#  "benczechmark_cs_court_decisions_ner"
#  "benczechmark_umimeto_qa"
#  "benczechmark_cermat_mc"
#  "benczechmark_cermat_qa"
#  "benczechmark_history_ir"
  "benczechmark_histcorpus" #requires logprob summing, not averaging!
  "benczechmark_essay"      #requires logprob summing, not averaging!
#  "benczechmark_fiction"    #requires logprob summing, not averaging!
)

# Define tasks that require summing of logprobs
SUM_LOGPROBS=(
  "benczechmark_histcorpus"
  "benczechmark_essay"
  "benczechmark_fiction"
)
