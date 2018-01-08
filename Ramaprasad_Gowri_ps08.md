Collaborators:
--------------

Nikhil Samtani, Neha Pothina

Data
----

I have explored credit card application data. This originates from a
confidential source, and all variable names are removed. The only
variable you have to know is A16: approval (+) or refusal (-). The data
is downloaded from [UCI Machine Learning
Repo](https://archive.ics.uci.edu/ml/datasets/Credit+Approval), more
information is in the meta file.

Task
----

I have tried to predict the approval or disapproval using logistic
regression and decision trees, and compare the performance of these
methods.

### 1. Select variables

Since I don't know the meaning of the variables,I have used
cross-tables, scatter plots, trial-and-error to find good predictors of
A16.

To examine, if the variable is a good predictor of A16, I have performed
the following:

1.  I have cleaned the data by filtering out '?' from the data.
2.  I have constructed crosstables for most catagorical variables and
    examined the p-value of the chi-square test, to test its reliability
    as a predictor.
3.  For numeric data, I have bucketed them to catagories and constructed
    crosstables to examine its fit.
4.  Finally for some of the variables I have made scatter-plots to
    examine the goodness of fit.

<!-- -->

    library("gmodels")
    library("dplyr")

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

    library("ggplot2")
    credit_score_data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data", header = FALSE, 
                                  col.names =
                                c("A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16") )
    str(credit_score_data)

    ## 'data.frame':    690 obs. of  16 variables:
    ##  $ A1 : Factor w/ 3 levels "?","a","b": 3 2 2 3 3 3 3 2 3 3 ...
    ##  $ A2 : Factor w/ 350 levels "?","13.75","15.17",..: 158 330 91 127 45 170 181 76 312 257 ...
    ##  $ A3 : num  0 4.46 0.5 1.54 5.62 ...
    ##  $ A4 : Factor w/ 4 levels "?","l","u","y": 3 3 3 3 3 3 3 3 4 4 ...
    ##  $ A5 : Factor w/ 4 levels "?","g","gg","p": 2 2 2 2 2 2 2 2 4 4 ...
    ##  $ A6 : Factor w/ 15 levels "?","aa","c","cc",..: 14 12 12 14 14 11 13 4 10 14 ...
    ##  $ A7 : Factor w/ 10 levels "?","bb","dd",..: 9 5 5 9 9 9 5 9 5 9 ...
    ##  $ A8 : num  1.25 3.04 1.5 3.75 1.71 ...
    ##  $ A9 : Factor w/ 2 levels "f","t": 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ A10: Factor w/ 2 levels "f","t": 2 2 1 2 1 1 1 1 1 1 ...
    ##  $ A11: int  1 6 0 5 0 0 0 0 0 0 ...
    ##  $ A12: Factor w/ 2 levels "f","t": 1 1 1 2 1 2 2 1 1 2 ...
    ##  $ A13: Factor w/ 3 levels "g","p","s": 1 1 1 1 3 1 1 1 1 1 ...
    ##  $ A14: Factor w/ 171 levels "?","00000","00017",..: 70 13 98 33 39 117 56 25 64 17 ...
    ##  $ A15: int  0 560 824 3 0 0 31285 1349 314 1442 ...
    ##  $ A16: Factor w/ 2 levels "-","+": 2 2 2 2 2 2 2 2 2 2 ...

    head(credit_score_data)

    ##   A1    A2    A3 A4 A5 A6 A7   A8 A9 A10 A11 A12 A13   A14 A15 A16
    ## 1  b 30.83 0.000  u  g  w  v 1.25  t   t   1   f   g 00202   0   +
    ## 2  a 58.67 4.460  u  g  q  h 3.04  t   t   6   f   g 00043 560   +
    ## 3  a 24.50 0.500  u  g  q  h 1.50  t   f   0   f   g 00280 824   +
    ## 4  b 27.83 1.540  u  g  w  v 3.75  t   t   5   t   g 00100   3   +
    ## 5  b 20.17 5.625  u  g  w  v 1.71  t   f   0   f   s 00120   0   +
    ## 6  b 32.08 4.000  u  g  m  v 2.50  t   f   0   t   g 00360   0   +

    tail(credit_score_data)

    ##     A1    A2     A3 A4 A5 A6 A7   A8 A9 A10 A11 A12 A13   A14 A15 A16
    ## 685  b 40.58  3.290  u  g  m  v 3.50  f   f   0   t   s 00400   0   -
    ## 686  b 21.08 10.085  y  p  e  h 1.25  f   f   0   f   g 00260   0   -
    ## 687  a 22.67  0.750  u  g  c  v 2.00  f   t   2   t   g 00200 394   -
    ## 688  a 25.25 13.500  y  p ff ff 2.00  f   t   1   t   g 00200   1   -
    ## 689  b 17.92  0.205  u  g aa  v 0.04  f   f   0   f   g 00280 750   -
    ## 690  b 35.00  3.375  u  g  c  h 8.29  f   f   0   t   g 00000   0   -

    #A1
    credit_score_data_A1_filtered <- credit_score_data %>% filter(credit_score_data$A1 != '?')
    crosstab_A1_A16 <- CrossTable(credit_score_data_A1_filtered$A16,credit_score_data_A1_filtered$A1, chisq = TRUE)

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## | Chi-square contribution |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  678 
    ## 
    ##  
    ##                                   | credit_score_data_A1_filtered$A1 
    ## credit_score_data_A1_filtered$A16 |         a |         b | Row Total | 
    ## ----------------------------------|-----------|-----------|-----------|
    ##                                 - |       112 |       262 |       374 | 
    ##                                   |     0.127 |     0.057 |           | 
    ##                                   |     0.299 |     0.701 |     0.552 | 
    ##                                   |     0.533 |     0.560 |           | 
    ##                                   |     0.165 |     0.386 |           | 
    ## ----------------------------------|-----------|-----------|-----------|
    ##                                 + |        98 |       206 |       304 | 
    ##                                   |     0.157 |     0.070 |           | 
    ##                                   |     0.322 |     0.678 |     0.448 | 
    ##                                   |     0.467 |     0.440 |           | 
    ##                                   |     0.145 |     0.304 |           | 
    ## ----------------------------------|-----------|-----------|-----------|
    ##                      Column Total |       210 |       468 |       678 | 
    ##                                   |     0.310 |     0.690 |           | 
    ## ----------------------------------|-----------|-----------|-----------|
    ## 
    ##  
    ## Statistics for All Table Factors
    ## 
    ## 
    ## Pearson's Chi-squared test 
    ## ------------------------------------------------------------
    ## Chi^2 =  0.4114351     d.f. =  1     p =  0.521242 
    ## 
    ## Pearson's Chi-squared test with Yates' continuity correction 
    ## ------------------------------------------------------------
    ## Chi^2 =  0.3112833     d.f. =  1     p =  0.5768938 
    ## 
    ## 

    #A2
    credit_score_data_A2_filtered <- credit_score_data %>%filter(credit_score_data$A2 != '?')
    credit_score_data_A2_filtered$A2 <- as.numeric(as.character(credit_score_data_A2_filtered$A2))
    credit_score_data_A2_filtered <- credit_score_data_A2_filtered %>% mutate(A16 = as.factor (ifelse (A16 == "+", "Plus","Minus")))
    ggplot(aes(A16,A2),data=credit_score_data_A2_filtered) + geom_boxplot()

![](Ramaprasad_Gowri_ps08_files/figure-markdown_strict/unnamed-chunk-1-1.png)

    #A3
    credit_score_data_A3_filtered <- credit_score_data %>% filter(credit_score_data$A3 != '?')
    credit_score_data_A3_filtered <- credit_score_data_A3_filtered %>% mutate(A16 = as.factor (ifelse (A16 == "+", "Plus","Minus")))
    ggplot(aes(A16,A3),data=credit_score_data_A3_filtered) + geom_boxplot()

![](Ramaprasad_Gowri_ps08_files/figure-markdown_strict/unnamed-chunk-1-2.png)

    #A4
    credit_score_data_A4_filtered <- credit_score_data %>% filter(credit_score_data$A4 != '?')
    crosstab_A4_A16 <- CrossTable(credit_score_data_A4_filtered$A16,credit_score_data_A4_filtered$A4, chisq = TRUE)

    ## Warning in chisq.test(t, correct = FALSE, ...): Chi-squared approximation
    ## may be incorrect

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## | Chi-square contribution |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  684 
    ## 
    ##  
    ##                                   | credit_score_data_A4_filtered$A4 
    ## credit_score_data_A4_filtered$A16 |         l |         u |         y | Row Total | 
    ## ----------------------------------|-----------|-----------|-----------|-----------|
    ##                                 - |         0 |       263 |       118 |       381 | 
    ##                                   |     1.114 |     2.355 |     8.152 |           | 
    ##                                   |     0.000 |     0.690 |     0.310 |     0.557 | 
    ##                                   |     0.000 |     0.507 |     0.724 |           | 
    ##                                   |     0.000 |     0.385 |     0.173 |           | 
    ## ----------------------------------|-----------|-----------|-----------|-----------|
    ##                                 + |         2 |       256 |        45 |       303 | 
    ##                                   |     1.401 |     2.961 |    10.251 |           | 
    ##                                   |     0.007 |     0.845 |     0.149 |     0.443 | 
    ##                                   |     1.000 |     0.493 |     0.276 |           | 
    ##                                   |     0.003 |     0.374 |     0.066 |           | 
    ## ----------------------------------|-----------|-----------|-----------|-----------|
    ##                      Column Total |         2 |       519 |       163 |       684 | 
    ##                                   |     0.003 |     0.759 |     0.238 |           | 
    ## ----------------------------------|-----------|-----------|-----------|-----------|
    ## 
    ##  
    ## Statistics for All Table Factors
    ## 
    ## 
    ## Pearson's Chi-squared test 
    ## ------------------------------------------------------------
    ## Chi^2 =  26.23407     d.f. =  2     p =  2.01068e-06 
    ## 
    ## 
    ## 

    #A5
    credit_score_data_A5_filtered <- credit_score_data %>% filter(credit_score_data$A5 != '?')
    crosstab_A5_A16 <- CrossTable(credit_score_data_A5_filtered$A16,credit_score_data_A5_filtered$A5, chisq = TRUE)

    ## Warning in chisq.test(t, correct = FALSE, ...): Chi-squared approximation
    ## may be incorrect

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## | Chi-square contribution |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  684 
    ## 
    ##  
    ##                                   | credit_score_data_A5_filtered$A5 
    ## credit_score_data_A5_filtered$A16 |         g |        gg |         p | Row Total | 
    ## ----------------------------------|-----------|-----------|-----------|-----------|
    ##                                 - |       263 |         0 |       118 |       381 | 
    ##                                   |     2.355 |     1.114 |     8.152 |           | 
    ##                                   |     0.690 |     0.000 |     0.310 |     0.557 | 
    ##                                   |     0.507 |     0.000 |     0.724 |           | 
    ##                                   |     0.385 |     0.000 |     0.173 |           | 
    ## ----------------------------------|-----------|-----------|-----------|-----------|
    ##                                 + |       256 |         2 |        45 |       303 | 
    ##                                   |     2.961 |     1.401 |    10.251 |           | 
    ##                                   |     0.845 |     0.007 |     0.149 |     0.443 | 
    ##                                   |     0.493 |     1.000 |     0.276 |           | 
    ##                                   |     0.374 |     0.003 |     0.066 |           | 
    ## ----------------------------------|-----------|-----------|-----------|-----------|
    ##                      Column Total |       519 |         2 |       163 |       684 | 
    ##                                   |     0.759 |     0.003 |     0.238 |           | 
    ## ----------------------------------|-----------|-----------|-----------|-----------|
    ## 
    ##  
    ## Statistics for All Table Factors
    ## 
    ## 
    ## Pearson's Chi-squared test 
    ## ------------------------------------------------------------
    ## Chi^2 =  26.23407     d.f. =  2     p =  2.01068e-06 
    ## 
    ## 
    ## 

    #A6
    credit_score_data_A6_filtered <- credit_score_data %>% filter(credit_score_data$A6 != '?')
    crosstab_A6_A16 <- CrossTable(credit_score_data_A6_filtered$A16,credit_score_data_A6_filtered$A6, chisq = TRUE)

    ## Warning in chisq.test(t, correct = FALSE, ...): Chi-squared approximation
    ## may be incorrect

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## | Chi-square contribution |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  681 
    ## 
    ##  
    ##                                   | credit_score_data_A6_filtered$A6 
    ## credit_score_data_A6_filtered$A16 |        aa |         c |        cc |         d |         e |        ff |         i |         j |         k |         m |         q |         r |         w |         x | Row Total | 
    ## ----------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
    ##                                 - |        35 |        75 |        12 |        23 |        11 |        46 |        45 |         7 |        37 |        22 |        27 |         1 |        31 |         6 |       378 | 
    ##                                   |     0.843 |     0.014 |     5.085 |     2.420 |     0.596 |     9.346 |     4.583 |     0.378 |     2.669 |     0.039 |     6.133 |     0.266 |     0.576 |    10.799 |           | 
    ##                                   |     0.093 |     0.198 |     0.032 |     0.061 |     0.029 |     0.122 |     0.119 |     0.019 |     0.098 |     0.058 |     0.071 |     0.003 |     0.082 |     0.016 |     0.555 | 
    ##                                   |     0.648 |     0.547 |     0.293 |     0.767 |     0.440 |     0.868 |     0.763 |     0.700 |     0.725 |     0.579 |     0.346 |     0.333 |     0.484 |     0.158 |           | 
    ##                                   |     0.051 |     0.110 |     0.018 |     0.034 |     0.016 |     0.068 |     0.066 |     0.010 |     0.054 |     0.032 |     0.040 |     0.001 |     0.046 |     0.009 |           | 
    ## ----------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
    ##                                 + |        19 |        62 |        29 |         7 |        14 |         7 |        14 |         3 |        14 |        16 |        51 |         2 |        33 |        32 |       303 | 
    ##                                   |     1.052 |     0.018 |     6.344 |     3.019 |     0.744 |    11.659 |     5.717 |     0.472 |     3.329 |     0.049 |     7.651 |     0.332 |     0.719 |    13.472 |           | 
    ##                                   |     0.063 |     0.205 |     0.096 |     0.023 |     0.046 |     0.023 |     0.046 |     0.010 |     0.046 |     0.053 |     0.168 |     0.007 |     0.109 |     0.106 |     0.445 | 
    ##                                   |     0.352 |     0.453 |     0.707 |     0.233 |     0.560 |     0.132 |     0.237 |     0.300 |     0.275 |     0.421 |     0.654 |     0.667 |     0.516 |     0.842 |           | 
    ##                                   |     0.028 |     0.091 |     0.043 |     0.010 |     0.021 |     0.010 |     0.021 |     0.004 |     0.021 |     0.023 |     0.075 |     0.003 |     0.048 |     0.047 |           | 
    ## ----------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
    ##                      Column Total |        54 |       137 |        41 |        30 |        25 |        53 |        59 |        10 |        51 |        38 |        78 |         3 |        64 |        38 |       681 | 
    ##                                   |     0.079 |     0.201 |     0.060 |     0.044 |     0.037 |     0.078 |     0.087 |     0.015 |     0.075 |     0.056 |     0.115 |     0.004 |     0.094 |     0.056 |           | 
    ## ----------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
    ## 
    ##  
    ## Statistics for All Table Factors
    ## 
    ## 
    ## Pearson's Chi-squared test 
    ## ------------------------------------------------------------
    ## Chi^2 =  98.3252     d.f. =  13     p =  3.49993e-15 
    ## 
    ## 
    ## 

    #A7
    credit_score_data_A7_filtered <- credit_score_data %>% filter(credit_score_data$A7 != '?')
    crosstab_A7_A16 <- CrossTable(credit_score_data_A7_filtered$A16,credit_score_data_A7_filtered$A7, chisq = TRUE)

    ## Warning in chisq.test(t, correct = FALSE, ...): Chi-squared approximation
    ## may be incorrect

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## | Chi-square contribution |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  681 
    ## 
    ##  
    ##                                   | credit_score_data_A7_filtered$A7 
    ## credit_score_data_A7_filtered$A16 |        bb |        dd |        ff |         h |         j |         n |         o |         v |         z | Row Total | 
    ## ----------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
    ##                                 - |        34 |         4 |        49 |        51 |         5 |         2 |         1 |       230 |         2 |       378 | 
    ##                                   |     0.048 |     0.135 |     9.527 |     8.555 |     0.070 |     0.022 |     0.011 |     0.328 |     1.341 |           | 
    ##                                   |     0.090 |     0.011 |     0.130 |     0.135 |     0.013 |     0.005 |     0.003 |     0.608 |     0.005 |     0.555 | 
    ##                                   |     0.576 |     0.667 |     0.860 |     0.370 |     0.625 |     0.500 |     0.500 |     0.576 |     0.250 |           | 
    ##                                   |     0.050 |     0.006 |     0.072 |     0.075 |     0.007 |     0.003 |     0.001 |     0.338 |     0.003 |           | 
    ## ----------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
    ##                                 + |        25 |         2 |         8 |        87 |         3 |         2 |         1 |       169 |         6 |       303 | 
    ##                                   |     0.060 |     0.168 |    11.885 |    10.673 |     0.088 |     0.027 |     0.014 |     0.410 |     1.673 |           | 
    ##                                   |     0.083 |     0.007 |     0.026 |     0.287 |     0.010 |     0.007 |     0.003 |     0.558 |     0.020 |     0.445 | 
    ##                                   |     0.424 |     0.333 |     0.140 |     0.630 |     0.375 |     0.500 |     0.500 |     0.424 |     0.750 |           | 
    ##                                   |     0.037 |     0.003 |     0.012 |     0.128 |     0.004 |     0.003 |     0.001 |     0.248 |     0.009 |           | 
    ## ----------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
    ##                      Column Total |        59 |         6 |        57 |       138 |         8 |         4 |         2 |       399 |         8 |       681 | 
    ##                                   |     0.087 |     0.009 |     0.084 |     0.203 |     0.012 |     0.006 |     0.003 |     0.586 |     0.012 |           | 
    ## ----------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
    ## 
    ##  
    ## Statistics for All Table Factors
    ## 
    ## 
    ## Pearson's Chi-squared test 
    ## ------------------------------------------------------------
    ## Chi^2 =  45.03421     d.f. =  8     p =  3.625453e-07 
    ## 
    ## 
    ## 

    #A8
    credit_score_data_A8_filtered <- credit_score_data %>% filter(credit_score_data$A8 != '?')
    credit_score_data_A8_filtered <- credit_score_data_A8_filtered %>% mutate(A16 = as.factor (ifelse (A16 == "+", "Plus","Minus")))
    ggplot(aes(A16,A8),data=credit_score_data_A8_filtered) + geom_boxplot()

![](Ramaprasad_Gowri_ps08_files/figure-markdown_strict/unnamed-chunk-1-3.png)

    #A9
    crosstab_A9_A16 <- CrossTable(credit_score_data$A16,credit_score_data$A9, chisq = TRUE)

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## | Chi-square contribution |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  690 
    ## 
    ##  
    ##                       | credit_score_data$A9 
    ## credit_score_data$A16 |         f |         t | Row Total | 
    ## ----------------------|-----------|-----------|-----------|
    ##                     - |       306 |        77 |       383 | 
    ##                       |    83.359 |    75.970 |           | 
    ##                       |     0.799 |     0.201 |     0.555 | 
    ##                       |     0.930 |     0.213 |           | 
    ##                       |     0.443 |     0.112 |           | 
    ## ----------------------|-----------|-----------|-----------|
    ##                     + |        23 |       284 |       307 | 
    ##                       |   103.995 |    94.777 |           | 
    ##                       |     0.075 |     0.925 |     0.445 | 
    ##                       |     0.070 |     0.787 |           | 
    ##                       |     0.033 |     0.412 |           | 
    ## ----------------------|-----------|-----------|-----------|
    ##          Column Total |       329 |       361 |       690 | 
    ##                       |     0.477 |     0.523 |           | 
    ## ----------------------|-----------|-----------|-----------|
    ## 
    ##  
    ## Statistics for All Table Factors
    ## 
    ## 
    ## Pearson's Chi-squared test 
    ## ------------------------------------------------------------
    ## Chi^2 =  358.1003     d.f. =  1     p =  7.29853e-80 
    ## 
    ## Pearson's Chi-squared test with Yates' continuity correction 
    ## ------------------------------------------------------------
    ## Chi^2 =  355.2038     d.f. =  1     p =  3.11859e-79 
    ## 
    ## 

    crosstab_A9_A16$chisq

    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  t
    ## X-squared = 358.1, df = 1, p-value < 2.2e-16

    #A10
    crosstab_A10_A16 <- CrossTable(credit_score_data$A16,credit_score_data$A10, chisq = TRUE)

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## | Chi-square contribution |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  690 
    ## 
    ##  
    ##                       | credit_score_data$A10 
    ## credit_score_data$A16 |         f |         t | Row Total | 
    ## ----------------------|-----------|-----------|-----------|
    ##                     - |       297 |        86 |       383 | 
    ##                       |    27.569 |    36.914 |           | 
    ##                       |     0.775 |     0.225 |     0.555 | 
    ##                       |     0.752 |     0.292 |           | 
    ##                       |     0.430 |     0.125 |           | 
    ## ----------------------|-----------|-----------|-----------|
    ##                     + |        98 |       209 |       307 | 
    ##                       |    34.393 |    46.052 |           | 
    ##                       |     0.319 |     0.681 |     0.445 | 
    ##                       |     0.248 |     0.708 |           | 
    ##                       |     0.142 |     0.303 |           | 
    ## ----------------------|-----------|-----------|-----------|
    ##          Column Total |       395 |       295 |       690 | 
    ##                       |     0.572 |     0.428 |           | 
    ## ----------------------|-----------|-----------|-----------|
    ## 
    ##  
    ## Statistics for All Table Factors
    ## 
    ## 
    ## Pearson's Chi-squared test 
    ## ------------------------------------------------------------
    ## Chi^2 =  144.9277     d.f. =  1     p =  2.227269e-33 
    ## 
    ## Pearson's Chi-squared test with Yates' continuity correction 
    ## ------------------------------------------------------------
    ## Chi^2 =  143.0696     d.f. =  1     p =  5.675727e-33 
    ## 
    ## 

    crosstab_A10_A16$chisq

    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  t
    ## X-squared = 144.93, df = 1, p-value < 2.2e-16

    #A11
    credit_score_data_A11_filtered <- credit_score_data %>% filter(credit_score_data$A11 != '?')
    credit_score_data_A11_filtered <- credit_score_data_A11_filtered %>% mutate(A16 = as.factor (ifelse (A16 == "+", "Plus","Minus")))
    ggplot(aes(A16,A11),data=credit_score_data_A11_filtered) + geom_boxplot()

![](Ramaprasad_Gowri_ps08_files/figure-markdown_strict/unnamed-chunk-1-4.png)

    #A12
    crosstab_A12_A16 <- CrossTable(credit_score_data$A16,credit_score_data$A12, chisq = TRUE)

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## | Chi-square contribution |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  690 
    ## 
    ##  
    ##                       | credit_score_data$A12 
    ## credit_score_data$A16 |         f |         t | Row Total | 
    ## ----------------------|-----------|-----------|-----------|
    ##                     - |       213 |       170 |       383 | 
    ##                       |     0.141 |     0.166 |           | 
    ##                       |     0.556 |     0.444 |     0.555 | 
    ##                       |     0.570 |     0.538 |           | 
    ##                       |     0.309 |     0.246 |           | 
    ## ----------------------|-----------|-----------|-----------|
    ##                     + |       161 |       146 |       307 | 
    ##                       |     0.175 |     0.208 |           | 
    ##                       |     0.524 |     0.476 |     0.445 | 
    ##                       |     0.430 |     0.462 |           | 
    ##                       |     0.233 |     0.212 |           | 
    ## ----------------------|-----------|-----------|-----------|
    ##          Column Total |       374 |       316 |       690 | 
    ##                       |     0.542 |     0.458 |           | 
    ## ----------------------|-----------|-----------|-----------|
    ## 
    ##  
    ## Statistics for All Table Factors
    ## 
    ## 
    ## Pearson's Chi-squared test 
    ## ------------------------------------------------------------
    ## Chi^2 =  0.6900889     d.f. =  1     p =  0.4061341 
    ## 
    ## Pearson's Chi-squared test with Yates' continuity correction 
    ## ------------------------------------------------------------
    ## Chi^2 =  0.5682733     d.f. =  1     p =  0.4509459 
    ## 
    ## 

    crosstab_A12_A16$chisq

    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  t
    ## X-squared = 0.69009, df = 1, p-value = 0.4061

    #A13
    crosstab_A13_A16 <- CrossTable(credit_score_data$A16,credit_score_data$A13, chisq = TRUE)

    ## Warning in chisq.test(t, correct = FALSE, ...): Chi-squared approximation
    ## may be incorrect

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## | Chi-square contribution |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  690 
    ## 
    ##  
    ##                       | credit_score_data$A13 
    ## credit_score_data$A16 |         g |         p |         s | Row Total | 
    ## ----------------------|-----------|-----------|-----------|-----------|
    ##                     - |       338 |         3 |        42 |       383 | 
    ##                       |     0.229 |     0.467 |     3.393 |           | 
    ##                       |     0.883 |     0.008 |     0.110 |     0.555 | 
    ##                       |     0.541 |     0.375 |     0.737 |           | 
    ##                       |     0.490 |     0.004 |     0.061 |           | 
    ## ----------------------|-----------|-----------|-----------|-----------|
    ##                     + |       287 |         5 |        15 |       307 | 
    ##                       |     0.286 |     0.583 |     4.233 |           | 
    ##                       |     0.935 |     0.016 |     0.049 |     0.445 | 
    ##                       |     0.459 |     0.625 |     0.263 |           | 
    ##                       |     0.416 |     0.007 |     0.022 |           | 
    ## ----------------------|-----------|-----------|-----------|-----------|
    ##          Column Total |       625 |         8 |        57 |       690 | 
    ##                       |     0.906 |     0.012 |     0.083 |           | 
    ## ----------------------|-----------|-----------|-----------|-----------|
    ## 
    ##  
    ## Statistics for All Table Factors
    ## 
    ## 
    ## Pearson's Chi-squared test 
    ## ------------------------------------------------------------
    ## Chi^2 =  9.19157     d.f. =  2     p =  0.01009429 
    ## 
    ## 
    ## 

    crosstab_A13_A16$chisq

    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  t
    ## X-squared = 9.1916, df = 2, p-value = 0.01009

    #A14

    credit_score_data_A14_filtered <- credit_score_data %>% filter(credit_score_data$A14 != '?')
    credit_score_data_A14_filtered <- credit_score_data_A14_filtered %>% mutate(A16 = as.factor (ifelse (A16 == "+", "Plus","Minus")))
    credit_score_data_A14_filtered$A14 <- as.numeric(as.character(credit_score_data_A14_filtered$A14))
    ggplot(aes(A16,A14),data=credit_score_data_A14_filtered) + geom_boxplot()

![](Ramaprasad_Gowri_ps08_files/figure-markdown_strict/unnamed-chunk-1-5.png)

    #A15
    credit_score_data_A15_filtered <- credit_score_data %>% filter(credit_score_data$A15 != '?')
    credit_score_data_A15_filtered <- credit_score_data_A15_filtered %>% mutate(A16 = as.factor (ifelse (A16 == "+", "Plus","Minus")))
    credit_score_data_A15_filtered$A15 <- as.numeric(as.character(credit_score_data_A15_filtered$A15))
    ggplot(aes(A16,A15),data=credit_score_data_A14_filtered) + geom_boxplot()

![](Ramaprasad_Gowri_ps08_files/figure-markdown_strict/unnamed-chunk-1-6.png)

    crosstab_A1_A16$chisq$p.value

    ## [1] 0.521242

    crosstab_A4_A16$chisq$p.value

    ## [1] 2.01068e-06

    crosstab_A5_A16$chisq$p.value

    ## [1] 2.01068e-06

    crosstab_A6_A16$chisq$p.value

    ## [1] 3.49993e-15

    crosstab_A7_A16$chisq$p.value

    ## [1] 3.625453e-07

    crosstab_A9_A16$chisq$p.value

    ## [1] 7.29853e-80

    crosstab_A10_A16$chisq$p.value

    ## [1] 2.227269e-33

    crosstab_A12_A16$chisq$p.value

    ## [1] 0.4061341

    crosstab_A13_A16$chisq$p.value

    ## [1] 0.01009429

Thus based on the tests, we can assess and that: 1. If the p-value is
significant (&gt;0.001) that it might be a good predictor. 2. Based on
this test the following variables can be considered significant: A4, A5,
A6,A7,A9,A10 3. For numeric variables, I have considered the box plot to
understand if there are outliers in the data and based on my
observations: a. A3: The mean for approved versus rejected is different
from each other. I thus would, use this for prediction. b. A8: The
means, seem to be similar and hence would discard it from the logistic
regression. c. A11: There is a significant difference in the means,and I
would consider this in the equation. d. A14: There does not seem to be a
significant difference in the means f. A15: The mean is not different,
and there seems to be outliers in the approved section and hence would
consider, it as a predictor.

### 2. Estimate logistic regression

I have used these variables to estimate logistic regression models. And
used this model to predict the outcome. I have made a cross-table of
actual/predicted outcomes.

Based on my intial analysis I am running the following logistic
regression model

    log_model <- glm(A16 ~ A4+A5+A6+A7+A9+A10+A3+A11+A15, family=binomial(link="logit"), data=credit_score_data)

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    predict_model<- predict(log_model, type="response") > 0.5
    table_predictions <- table(credit_score_data$A16,predict_model)
    table_predictions

    ##    predict_model
    ##     FALSE TRUE
    ##   -   328   55
    ##   +    27  280

    correct_predictions <- table_predictions %>% diag() %>% sum()
    total_predictions <- table_predictions %>% sum()
    accuracy_predictions <- correct_predictions/total_predictions
    accuracy_predictions

    ## [1] 0.8811594

This model gives an 88.12% accuracy.

### 3. Estimate decision trees.

I have used the same variables to compute decision tree models. As
above, I predict the result, by making a cross-table, and find the
correct percentage.

    library(rpart)
    tree_model <- rpart(A16 ~ A4+A5+A6+A7+A9+A10+A3+A11+A15,data=credit_score_data)
    predict_tree_model <- predict(tree_model, type = "class")
    table_tree_model <- table(credit_score_data$A16, predict_tree_model)
    table_tree_model

    ##    predict_tree_model
    ##       -   +
    ##   - 341  42
    ##   +  38 269

    correct_tree_model_predictions <- table_tree_model %>% diag() %>% sum()
    total_tree_model_predictions <- table_tree_model %>% sum()
    accuracy_tree_model <- correct_tree_model_predictions/total_tree_model_predictions
    accuracy_tree_model

    ## [1] 0.884058

The accuracy of this model is 88.41%

### 4. Repeat the process

I have repeated this process with different variables Model 1:

    log_model_1 <- glm(A16 ~ A4+A5+A6+A7+A9+A10, family=binomial(link="logit"), data=credit_score_data)

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    predict_model_1 <- predict(log_model_1, type="response") > 0.5
    table_predictions_1 <- table(credit_score_data$A16,predict_model_1)
    table_predictions_1

    ##    predict_model_1
    ##     FALSE TRUE
    ##   -   325   58
    ##   +    23  284

    correct_predictions_1 <- table_predictions_1 %>% diag() %>% sum()
    total_predictions_1 <- table_predictions_1 %>% sum()
    accuracy_predictions_1 <- correct_predictions_1/total_predictions_1
    accuracy_predictions_1

    ## [1] 0.8826087

    tree_model_1 <- rpart(A16 ~ A4+A5+A6+A7+A9+A10,data=credit_score_data)
    predict_tree_model_1 <- predict(tree_model_1, type = "class")
    table_tree_model_1 <- table(credit_score_data$A16, predict_tree_model_1)
    table_tree_model_1

    ##    predict_tree_model_1
    ##       -   +
    ##   - 341  42
    ##   +  38 269

    correct_tree_model_predictions_1 <- table_tree_model_1 %>% diag() %>% sum()
    total_tree_model_predictions_1 <- table_tree_model_1 %>% sum()
    accuracy_tree_model_1 <- correct_tree_model_predictions_1/total_tree_model_predictions_1
    accuracy_tree_model_1

    ## [1] 0.884058

Model 2:

    log_model_2 <- glm(A16 ~ A3+A11+A15, family=binomial(link="logit"), data=credit_score_data)

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    predict_model_2 <- predict(log_model_2, type="response") > 0.5
    table_predictions_2 <- table(credit_score_data$A16,predict_model_2)
    table_predictions_2

    ##    predict_model_2
    ##     FALSE TRUE
    ##   -   352   31
    ##   +   131  176

    correct_predictions_2 <- table_predictions_2 %>% diag() %>% sum()
    total_predictions_2 <- table_predictions_2 %>% sum()
    accuracy_predictions_2 <- correct_predictions_2/total_predictions_2
    accuracy_predictions_2

    ## [1] 0.7652174

    tree_model_2 <- rpart(A16 ~ A4+A5+A6+A7+A9+A10,data=credit_score_data)
    predict_tree_model_2 <- predict(tree_model_2, type = "class")
    table_tree_model_2 <- table(credit_score_data$A16, predict_tree_model_2)
    table_tree_model_2

    ##    predict_tree_model_2
    ##       -   +
    ##   - 341  42
    ##   +  38 269

    correct_tree_model_predictions_2 <- table_tree_model_2 %>% diag() %>% sum()
    total_tree_model_predictions_2 <- table_tree_model_2 %>% sum()
    accuracy_tree_model_2 <- correct_tree_model_predictions_2/total_tree_model_predictions_2
    accuracy_tree_model_2

    ## [1] 0.884058

Model 3:

    log_model_3 <- glm(A16 ~ A4+A5+A6+A7+A9+A10+A2+A13, family=binomial(link="logit"), data=credit_score_data)

    ## Warning: glm.fit: algorithm did not converge

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    predict_model_3 <- predict(log_model_3, type="response") > 0.5
    table_predictions_3 <- table(credit_score_data$A16,predict_model_3)
    table_predictions_3

    ##    predict_model_3
    ##     FALSE TRUE
    ##   -   373   10
    ##   +    16  291

    correct_predictions_3 <- table_predictions_3 %>% diag() %>% sum()
    total_predictions_3 <- table_predictions_3 %>% sum()
    accuracy_predictions_3 <- correct_predictions_3/total_predictions_3
    accuracy_predictions_3

    ## [1] 0.9623188

    tree_model_3 <- rpart(A16 ~ A4+A5+A6+A7+A9+A10+A2+A13,data=credit_score_data)
    predict_tree_model_3 <- predict(tree_model_3, type = "class")
    table_tree_model_3 <- table(credit_score_data$A16, predict_tree_model_3)
    table_tree_model_3

    ##    predict_tree_model_3
    ##       -   +
    ##   - 371  12
    ##   +   8 299

    correct_tree_model_predictions_3 <- table_tree_model_3 %>% diag() %>% sum()
    total_tree_model_predictions_3 <- table_tree_model_3 %>% sum()
    accuracy_tree_model_3 <- correct_tree_model_predictions_3/total_tree_model_predictions_3
    accuracy_tree_model_3

    ## [1] 0.9710145

### 5. Compare the models

From the above analysis, decision tree seems to have a higher degree of
precision than logistic regression. The best fit was variables
A4+A5+A6+A7+A9+A10.
