
#Normalized Weighted Gini
#https://www.kaggle.com/c/liberty-mutual-fire-peril/discussion/9880
#http://blog.nguyenvq.com/blog/2015/09/25/calculate-the-weighted-gini-coefficient-or-auc-in-r/
WeightedGini <- function(act, pred, weights)
{ df = data.frame(act = act, weights = weights, pred = pred) 
  df <- df[order(df$pred, decreasing = TRUE),] 
  df$random = cumsum((df$weights/sum(df$weights))) 
    totalPositive <- sum(df$act * df$weights) 
    df$cumPosFound <- cumsum(df$act * df$weights) 
      df$Lorentz <- df$cumPosFound / totalPositive 
      n <- nrow(df) 
        sum(df$Lorentz[-1]*df$random[-n]) - sum(df$Lorentz[-n]*df$random[-1]) 
}

NormalizedWeightedGini <- function(act, pred, weights)
{
	    WeightedGini(act, pred, weights) / WeightedGini(act, act, weights)
}
# a test case:

#var11 <- c(1, 2, 5, 4, 3) 
#pred <- c(0.1, 0.4, 0.3, 1.2, 0.0) 
#act <- c(0, 0, 1, 0, 1)

#should now score -0.821428571428572.

