library(tidyverse)
library(survey)
library(broom)
library(cobalt)
library(WeightIt)
library(tableone)
library(survival)
library(prodlim)
library(survminer)
library(gtsummary)
library(ebal)
library(forestplot)
library(grid)
library(gridExtra)
library(gt)
library(patchwork)

create_weights <- function(filter_table,title){   

df <- SparkR::collect(filter_table)

# CREATE WEIGHTS
## Using weighit packaage perform entropy balancing with estimand of ATT

w <- weightit(cohort ~ weight + height + bmi + ace_before + arb_before + statin_before + anticoagulant_before + aspirin_before + cad + cancer + ckd123 + heart_failure + hypertension + liver_disease + a1c_gt65_before + a1c_last + nafld_before + adjustment_disorder_before + dysthymia_before + seasonal_affective_disorder_before + hypomania_before + dysphoria_before + insomnia_before + bipolar_before + mood_disorder_before + steroids_before + creatinine_before_mean + Harmonized_age_group + Harmonized_gender + race_ethnicity + binge_eating_disorder_before + bulimia_nervosa_before + generalized_anxiety_before + major_depression_disorder_before + ocd_before + panic_disorder_before + ptsd_before + social_anxiety_disorder_before + asthma_before + copd_before + bronchiolitis_obliterans_before + eosinophilic_esophagitis_before + phct_before + mast_cell_activation_before + exercise_induced_asthma_before + exercise_induced_bronchoconstriction_before + allergic_rhinitis_before + prediabetes_before + t1dm_before + t2dm_before + gestational_dm_before + antipsychotic_induced_weight_gain_before + pcos_before + 
#ckd4_before + esrd_dialysis_before + 
biguanides_before_365 + dpp4_before_365 + glp1_before_365 + sglt2_before_365 + su_before_365 + thiazo_before_365 + insulin_before_365 + other_diabetes_med_before_365 + egfr +
data_partner_id ,
data = df, missing="ind",estimand="ATT", method="ebal") 
   
## Add weights to dataframe
df$weights <- w$weights
  
## look at summary of weighit object  
print(w)
print("summary: weighit")
summary(w)

# VISUALIZE BALANCE  
## make a love plot of smd before and after weighting
balance <- bal.tab(w, stats = c("m", "v"), thresholds = c(m = .1), un=T, continuous="std")

print("bal tab")
print(balance)
   new.names <- c(weight = "Weight",
height = "Height",
bmi = "BMI",
ace_before = "ACEi",
arb_before  = "ARB",
statin_before = "Statins ",
anticoagulant_before  = "Anticoagulant",
aspirin_before = "aspirin ",
steroids_before = "Steroids", 
cad = "CAD",
cancer = "Cancer",
ckd123 = "CKD, Stage 1-3",
heart_failure = "Heart Failure",
hypertension = "Hypertension",
liver_disease = "Liver Disease",
a1c_gt65_before = "HbA1c>=6.5",
a1c_last = "Last HbA1c ",
nafld_before = "NAFLD",
adjustment_disorder_before = "Adjustment d/o",
dysthymia_before = "Dysthymia",
seasonal_affective_disorder_before = "SAD",
hypomania_before = "Hypomania",
dysphoria_before = "Dysphoria",
insomnia_before = "Insomnia ",
bipolar_before = "Bipolar Disease ",
mood_disorder_before = "Mood Disorder",
creatinine_before_mean = "Creatinine",
Harmonized_age_group = "Age",
Harmonized_gender = "Gender ",
race_ethnicity = "Race, Ethnicity ",
binge_eating_disorder_before = "Binge eating d/o",
bulimia_nervosa_before = "Bulimia",
generalized_anxiety_before = "GAD",
major_depression_disorder_before = "Depression",
ocd_before = "OCD",
panic_disorder_before = "Panic d/o ",
ptsd_before = "Post Traumatic Stress d/o ",
social_anxiety_disorder_before = "Social Anxiety d/o",
asthma_before = "Asthma",
copd_before = "COPD",
bronchiolitis_obliterans_before = "Bronchiolitis obliterans",
eosinophilic_esophagitis_before = "Eosinophilic esophagitis",
phct_before = "Post-hematopoietic cell transplantation",
mast_cell_activation_before = "Mast Cell Activation",
exercise_induced_asthma_before = "Exercised-induced Asthma",
exercise_induced_bronchoconstriction_before = "Exercised-induced bronchoconstriction",
allergic_rhinitis_before = "Allergic rhinitis",
prediabetes_before = "Prediabetes",
t1dm_before = "Type 1 DM",
t2dm_before = "Type 2 DM",
gestational_dm_before = "Gestational DM",
antipsychotic_induced_weight_gain_before = "Antipsychotic weight gain",
pcos_before = "Polycystic Ovarian Syndrome",
biguanides_before_365 = "Biguanides",
dpp4_before_365 = "DPP4i ",
glp1_before_365 = "GLP-1 RA",
sglt2_before_365 = "SGLT-2 inhibitor",
su_before_365 = "Sulfonylureas",
thiazo_before_365 = "Thiazolidinediones",
insulin_before_365 = "Outpatient Insulin",
other_diabetes_med_before_365 = "Other Diabetes meds ",
egfr = "estimated GFR",
data_partner_id = "Data Partner"
   )

lp <- love.plot(balance, var.names=new.names, title = title)
print(lp)

## check distributions of key continuous variables to ensure balance 
a1c <- bal.plot(w, "a1c_last", which = "both", type = "ecdf")
print(a1c)

a1c2 <- bal.plot(w, "a1c_last", which = "both")
print(a1c2)

weightp <- bal.plot(w, "weight", which = "both", type = "ecdf")
print(weightp)

weightp2 <- bal.plot(w, "weight", which = "both")
print(weightp2)

bmip <- bal.plot(w, "bmi", which = "both", type = "ecdf")
print(bmip)
bmip2 <- bal.plot(w, "bmi", which = "both")
print(bmip2)

t2dmp2 <- bal.plot(w, "t2dm_before", which = "both")
print(t2dmp2)

return(df)
}

create_analysis <- function(final_df,start_days,end_days) {
  df <- SparkR::select(final_df,c("weights","cohort","combined_outcome", "combined_time_to_censor","combined_censor_type")) %>% 
    SparkR::collect()

# CREATE SURVEY DESIGN
   data_svy <- svydesign(data = df, weights = ~weights, ids = ~ 1)
   print(summary(data_svy))

# LOGISTIC REGRESSION
   glm <- survey::svyglm(combined_outcome~cohort,design=data_svy, family=quasibinomial())
   print(summary(glm))
   CI_glm <- as.data.frame(confint(glm)) %>% setNames(c("conf.low", "conf.high"))

   tb_glm <- tidy(glm)
   df_glm <-cbind(analysis = "glm", tb_glm, CI_glm, robust.se = NA, cohort= NA, time = NA, n.risk = NA, n.event = NA, n.lost = NA, surv = NA, se.surv = NA, lower = NA, upper = NA)  

# COXPH
    cox <- svycoxph(Surv(combined_time_to_censor, combined_outcome) ~cohort, design=data_svy)
    testph <- cox.zph(cox)
    print("Proportional Hazards Test")
    print(testph)

    tb_cox <- tidy(cox)
    df_cox <-cbind(analysis = "coxph", tb_cox, cohort= NA, time = NA, n.risk = NA, n.event = NA, n.lost = NA, surv = NA, se.surv = NA, lower = NA, upper = NA) 

   
# PRODLIM 

crFit <- prodlim::prodlim(Hist(combined_time_to_censor,combined_censor_type, cens.code = "censored")~cohort, data = df, caseweights = df$weights)
summary(crFit)
print(crFit$model.response)
summ_list <- summary(crFit,times=c(0,30,60,90,120,150,180),cause="outcome")
summ_df<- as.data.frame(summ_list$table[[1]])
res <- cbind(cohort ="control", summ_df)
summ_df2<- as.data.frame(summ_list$table[[2]])
res2 <- cbind(cohort ="metformin", summ_df2)
res3 <- rbind(res, res2)
print(summ_list)
print(res3)

df_prodlim <-cbind(analysis = "prodlim", res3, term = NA, estimate = NA, std.error = NA, statistic = NA, p.value = NA, conf.low = NA, conf.high = NA, robust.se = NA) 

par(cex.axis=1.25,cex.lab=1.25, cex=1.25, mar=c(10,10,10,10))
plot(crFit, type="risk", cause="outcome",ylim=c(0, .15), xlim=c(0,180), legend = F, xlab="", ylab="", atrisk=F, axis1.at=seq(0,180,30),
       axis1.labels=seq(0,180,30),col =c("#F15C52", "#5DC0D8"))
 mtext("Cumulative Incidence (%)", side = 2, line = 4.5, cex=2)
  mtext("Time (days)", side = 1, line = 4.5, cex=2)
legend("topleft", legend = c("comparator", "metformin"),col = c("#F15C52","#5DC0D8"), bty = "n", pch = 15, pt.cex = 5, yjust=0, y.intersp = 0.25, x.intersp = 0.25)

# COMBINE RESULT OUTPUT 
result_df <- rbind(df_glm, df_cox, df_prodlim)  
return(result_df)
}

create_analysis_w_int <- function(final_df,start_days,end_days) {
  df <- SparkR::select(final_df,c("weights","cohort","combined_outcome", "combined_time_to_censor","combined_censor_type","combined_days_from_index")) %>% 
    SparkR::collect()

# CREATE SURVEY DESIGN
   data_svy <- svydesign(data = df, weights = ~weights, ids = ~ 1)
   print(summary(data_svy))

# LOGISTIC REGRESSION
   glm <- survey::svyglm(combined_outcome~cohort + combined_days_from_index + cohort*combined_days_from_index,design=data_svy, family=quasibinomial())
   print(summary(glm))
   CI_glm <- as.data.frame(confint(glm)) %>% setNames(c("conf.low", "conf.high"))

   tb_glm <- tidy(glm)
   df_glm <-cbind(analysis = "glm", tb_glm, CI_glm, robust.se = NA, cohort= NA, time = NA, n.risk = NA, n.event = NA, n.lost = NA, surv = NA, se.surv = NA, lower = NA, upper = NA)  

# COXPH
    cox <- svycoxph(Surv(combined_time_to_censor, combined_outcome) ~cohort, design=data_svy)
    testph <- cox.zph(cox)
    print("Proportional Hazards Test")
    print(testph)

    tb_cox <- tidy(cox)
    df_cox <-cbind(analysis = "coxph", tb_cox, cohort= NA, time = NA, n.risk = NA, n.event = NA, n.lost = NA, surv = NA, se.surv = NA, lower = NA, upper = NA) 

   
# PRODLIM 

crFit <- prodlim::prodlim(Hist(combined_time_to_censor,combined_censor_type, cens.code = "censored")~cohort, data = df, caseweights = df$weights)
summary(crFit)
print(crFit$model.response)
summ_list <- summary(crFit,times=c(0,30,60,90,120,150,180),cause="outcome")
summ_df<- as.data.frame(summ_list$table[[1]])
res <- cbind(cohort ="control", summ_df)
summ_df2<- as.data.frame(summ_list$table[[2]])
res2 <- cbind(cohort ="metformin", summ_df2)
res3 <- rbind(res, res2)
print(summ_list)
print(res3)

df_prodlim <-cbind(analysis = "prodlim", res3, term = NA, estimate = NA, std.error = NA, statistic = NA, p.value = NA, conf.low = NA, conf.high = NA, robust.se = NA) 

par(cex.axis=1.25,cex.lab=1.25, cex=1.25, mar=c(10,10,10,10))
plot(crFit, type="risk", cause="outcome",ylim=c(0, .15), xlim=c(0,180), legend = F, xlab="", ylab="", atrisk=F, axis1.at=seq(0,180,30),
       axis1.labels=seq(0,180,30))
 mtext("Cumulative Incidence (%)", side = 2, line = 4.5, cex=2)
  mtext("Time (days)", side = 1, line = 4.5, cex=2)
legend("topleft", legend = c("comparator", "metformin"),fill = c("black","darkgoldenrod2"), lty = c(1,1), bty = "n")

# COMBINE RESULT OUTPUT 
result_df <- rbind(df_glm, df_cox, df_prodlim)  
return(result_df)
}

create_weights_for_exc_dm <- function(filter_table, title){   

df <- SparkR::collect(filter_table)

# CREATE WEIGHTS
## Using weighit packaage perform entropy balancing with estimand of ATT

w <- weightit(cohort ~ weight + height + bmi + ace_before + arb_before + statin_before + anticoagulant_before + aspirin_before + cad + cancer + ckd123 + heart_failure + hypertension + liver_disease + a1c_gt65_before + a1c_last + nafld_before + adjustment_disorder_before + dysthymia_before + seasonal_affective_disorder_before + hypomania_before + dysphoria_before + insomnia_before + bipolar_before + mood_disorder_before + steroids_before + creatinine_before_mean + Harmonized_age_group + Harmonized_gender + race_ethnicity + binge_eating_disorder_before + bulimia_nervosa_before + generalized_anxiety_before + major_depression_disorder_before + ocd_before + panic_disorder_before + ptsd_before + social_anxiety_disorder_before + asthma_before + copd_before + bronchiolitis_obliterans_before + eosinophilic_esophagitis_before + phct_before + mast_cell_activation_before + exercise_induced_asthma_before + exercise_induced_bronchoconstriction_before + allergic_rhinitis_before + #prediabetes_before + t1dm_before + t2dm_before + gestational_dm_before + antipsychotic_induced_weight_gain_before + pcos_before + #ckd4_before + esrd_dialysis_before + 
biguanides_before_365 + dpp4_before_365 + glp1_before_365 + sglt2_before_365 + su_before_365 + thiazo_before_365 + insulin_before_365 + other_diabetes_med_before_365 + egfr +
data_partner_id ,
data = df, missing="ind",estimand="ATT", method="ebal") 
   
## Add weights to dataframe
df$weights <- w$weights
  
## look at summary of weighit object  
print(w)
print("summary: weighit")
summary(w)

# VISUALIZE BALANCE  
## make a love plot of smd before and after weighting
balance <- bal.tab(w, stats = c("m", "v"), thresholds = c(m = .1), un=T, continuous="std")

print("bal tab")
print(balance)
   new.names <- c(weight = "Weight",
height = "Height",
bmi = "BMI",
ace_before = "ACEi",
arb_before  = "ARB",
statin_before = "Statins ",
anticoagulant_before  = "Anticoagulant",
aspirin_before = "aspirin ",
steroids_before = "Steroids", 
cad = "CAD",
cancer = "Cancer",
ckd123 = "CKD, Stage 1-3",
heart_failure = "Heart Failure",
hypertension = "Hypertension",
liver_disease = "Liver Disease",
a1c_gt65_before = "HbA1c>=6.5",
a1c_last = "Last HbA1c ",
nafld_before = "NAFLD",
adjustment_disorder_before = "Adjustment d/o",
dysthymia_before = "Dysthymia",
seasonal_affective_disorder_before = "SAD",
hypomania_before = "Hypomania",
dysphoria_before = "Dysphoria",
insomnia_before = "Insomnia ",
bipolar_before = "Bipolar Disease ",
mood_disorder_before = "Mood Disorder",
creatinine_before_mean = "Creatinine",
Harmonized_age_group = "Age",
Harmonized_gender = "Gender ",
race_ethnicity = "Race, Ethnicity ",
binge_eating_disorder_before = "Binge eating d/o",
bulimia_nervosa_before = "Bulimia",
generalized_anxiety_before = "GAD",
major_depression_disorder_before = "Depression",
ocd_before = "OCD",
panic_disorder_before = "Panic d/o ",
ptsd_before = "Post Traumatic Stress d/o ",
social_anxiety_disorder_before = "Social Anxiety d/o",
asthma_before = "Asthma",
copd_before = "COPD",
bronchiolitis_obliterans_before = "Bronchiolitis obliterans",
eosinophilic_esophagitis_before = "Eosinophilic esophagitis",
phct_before = "Post-hematopoietic cell transplantation",
mast_cell_activation_before = "Mast Cell Activation",
exercise_induced_asthma_before = "Exercised-induced Asthma",
exercise_induced_bronchoconstriction_before = "Exercised-induced bronchoconstriction",
allergic_rhinitis_before = "Allergic rhinitis",
biguanides_before_365 = "Biguanides",
dpp4_before_365 = "DPP4i ",
glp1_before_365 = "GLP-1 RA",
sglt2_before_365 = "SGLT-2 inhibitor",
su_before_365 = "Sulfonylureas",
thiazo_before_365 = "Thiazolidinediones",
insulin_before_365 = "Outpatient Insulin",
other_diabetes_med_before_365 = "Other Diabetes meds ",
egfr = "estimated GFR",
data_partner_id = "Data Partner"
   )
lp <- love.plot(balance, var.names=new.names, title = title)

print(lp)

## check distributions of key continuous variables to ensure balance 
a1c <- bal.plot(w, "a1c_last", which = "both", type = "ecdf")
print(a1c)

a1c2 <- bal.plot(w, "a1c_last", which = "both")
print(a1c2)

weightp <- bal.plot(w, "weight", which = "both", type = "ecdf")
print(weightp)

weightp2 <- bal.plot(w, "weight", which = "both")
print(weightp2)

bmip <- bal.plot(w, "bmi", which = "both", type = "ecdf")
print(bmip)
bmip2 <- bal.plot(w, "bmi", which = "both")
print(bmip2)

return(df)
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.927489b6-d598-430f-bfc5-e70cb0d08f77"),
    weighted_exc_dm_0_to_1=Input(rid="ri.foundry.main.dataset.858b7fe8-5f0c-40fd-87d3-e62647da0751")
)
analysis_exc_dm_0_to_1 <- function( weighted_exc_dm_0_to_1) {

result_df <- create_analysis(weighted_exc_dm_0_to_1)
return(result_df)
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c244a361-028e-486d-a235-b5ce93205ccf"),
    weighted_exc_dm_0_to_14=Input(rid="ri.foundry.main.dataset.da1d4195-4ba3-4701-ba41-9e0762505dcd")
)
analysis_exc_dm_0_to_14 <- function( weighted_exc_dm_0_to_14) {

result_df <- create_analysis(weighted_exc_dm_0_to_14)
return(result_df)
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.5761aa39-71fc-479d-9a85-adb0da0e9a98"),
    weighted_exc_dm_0_to_6=Input(rid="ri.foundry.main.dataset.de8ad4b6-6fa7-4fda-8c20-5a2128ff96d7")
)
analysis_exc_dm_0_to_6 <- function( weighted_exc_dm_0_to_6) {

  df <- SparkR::select(weighted_exc_dm_0_to_6,c("weights","cohort","combined_outcome", "combined_time_to_censor","combined_censor_type")) %>% 
    SparkR::collect()

# CREATE SURVEY DESIGN
   data_svy <- svydesign(data = df, weights = ~weights, ids = ~ 1)
   print(summary(data_svy))

# LOGISTIC REGRESSION
   glm <- survey::svyglm(combined_outcome~cohort,design=data_svy, family=quasibinomial())
   print(summary(glm))
   CI_glm <- as.data.frame(confint(glm)) %>% setNames(c("conf.low", "conf.high"))

   tb_glm <- tidy(glm)
   df_glm <-cbind(analysis = "glm", tb_glm, CI_glm, robust.se = NA, cohort= NA, time = NA, n.risk = NA, n.event = NA, n.lost = NA, surv = NA, se.surv = NA, lower = NA, upper = NA)  

# COXPH
    cox <- svycoxph(Surv(combined_time_to_censor, combined_outcome) ~cohort, design=data_svy)
    testph <- cox.zph(cox)
    print("Proportional Hazards Test")
    print(testph)

    tb_cox <- tidy(cox)
    df_cox <-cbind(analysis = "coxph", tb_cox, cohort= NA, time = NA, n.risk = NA, n.event = NA, n.lost = NA, surv = NA, se.surv = NA, lower = NA, upper = NA) 

   
# PRODLIM 

crFit <- prodlim::prodlim(Hist(combined_time_to_censor,combined_censor_type, cens.code = "censored")~cohort, data = df, caseweights = df$weights)
summary(crFit)
print(crFit$model.response)
summ_list <- summary(crFit,times=c(0,30,60,90,120,150,180),cause="outcome")
summ_df<- as.data.frame(summ_list$table[[1]])
res <- cbind(cohort ="control", summ_df)
summ_df2<- as.data.frame(summ_list$table[[2]])
res2 <- cbind(cohort ="metformin", summ_df2)
res3 <- rbind(res, res2)
print(summ_list)
print(res3)

df_prodlim <-cbind(analysis = "prodlim", res3, term = NA, estimate = NA, std.error = NA, statistic = NA, p.value = NA, conf.low = NA, conf.high = NA, robust.se = NA) 

par(cex.axis=1.25,cex.lab=1.25, cex=1.25, mar=c(10,10,10,10))
plot(crFit, type="risk", cause="outcome",ylim=c(0, .15), xlim=c(0,180), legend = F, xlab="", ylab="", atrisk=F, axis1.at=seq(0,180,30),
       axis1.labels=seq(0,180,30),col =c("#F15C52", "#5DC0D8"))
 mtext("Cumulative Incidence (%)", side = 2, line = 4.5, cex=2)
  mtext("Time (days)", side = 1, line = 4.5, cex=2)
legend("topleft", legend = c("comparator", "metformin"),col = c("#F15C52","#5DC0D8"), bty = "n", pch = 15, pt.cex = 5, yjust=0, y.intersp = 0.25, x.intersp = 0.25)

# COMBINE RESULT OUTPUT 
result_df <- rbind(df_glm, df_cox, df_prodlim)  
return(result_df)
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.1788cb60-ea0b-4fa5-92ee-5372ac2e2913"),
    weighted_exc_dm_7_to_14=Input(rid="ri.foundry.main.dataset.3a99f2f9-f4f1-452b-b46f-3cade99093ac")
)
analysis_exc_dm_7_to_14 <- function( weighted_exc_dm_7_to_14) {

result_df <- create_analysis(weighted_exc_dm_7_to_14)
return(result_df)
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a6f4a2ac-4439-479b-bb3a-e26936e8145e"),
    weighted_exc_fluv_0_to_14=Input(rid="ri.foundry.main.dataset.13f25daf-2333-47c3-945b-7f3c9a50dc47")
)
analysis_exc_fluv_0_to_14 <- function(weighted_exc_fluv_0_to_14) {
 
result_df <- create_analysis(weighted_exc_fluv_0_to_14)
return(result_df)      
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.85aa4d90-a3a1-456d-aca7-30e7b788590f"),
    weighted_exc_fluv_0_to_6=Input(rid="ri.foundry.main.dataset.98e56d58-abc1-43d0-8969-6029af8a416c")
)
analysis_exc_fluv_0_to_6 <- function(weighted_exc_fluv_0_to_6) {
 
result_df <- create_analysis(weighted_exc_fluv_0_to_6)
return(result_df)   
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c09acbfc-435d-4d4c-9262-2cbb1f00d988"),
    weighted_exc_iver_0_to_14=Input(rid="ri.foundry.main.dataset.1e41bf28-4dd2-461c-b56f-8229ffec2c54")
)
analysis_exc_iver_0_to_14 <- function(weighted_exc_iver_0_to_14) {
  
result_df <- create_analysis(weighted_exc_iver_0_to_14)
return(result_df)      
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a487ee0c-e5f5-4620-bb43-d9856e557761"),
    weighted_exc_iver_0_to_6=Input(rid="ri.foundry.main.dataset.66bd84f9-fbce-497e-954d-256be5d9facb")
)
analysis_exc_iver_0_to_6 <- function(weighted_exc_iver_0_to_6) {
  
result_df <- create_analysis(weighted_exc_iver_0_to_6)
return(result_df)    
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.029e5ad8-8a2b-42ff-8305-d6755e4cedfb"),
    weighted_only_fluv_0_to_14=Input(rid="ri.foundry.main.dataset.fbfb30e9-b6ce-45a7-9454-61c539ee4d6b")
)
analysis_only_fluv_0_to_14 <- function(weighted_only_fluv_0_to_14) {
 
result_df <- create_analysis(weighted_only_fluv_0_to_14)
return(result_df)      
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0e1494ad-d2a5-4521-affb-eec1cd9697fd"),
    weighted_only_fluv_0_to_6=Input(rid="ri.foundry.main.dataset.80fe94fc-0d6a-4d95-9cdb-582f67dab25f")
)
analysis_only_fluv_0_to_6 <- function(weighted_only_fluv_0_to_6) {
 
result_df <- create_analysis(weighted_only_fluv_0_to_6)
return(result_df)      
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.bf9b2fb9-017b-4ade-8879-50a8864661ba"),
    weighted_only_iver_0_to_14=Input(rid="ri.foundry.main.dataset.888bf6bb-87bb-411f-bcbf-c38a2ee90e16")
)
analysis_only_iver_0_to_14 <- function(weighted_only_iver_0_to_14) {
   
result_df <- create_analysis(weighted_only_iver_0_to_14)
return(result_df)      
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.01fa85c3-52a2-4064-aa86-288ac08058b3"),
    weighted_only_iver_0_to_6=Input(rid="ri.foundry.main.dataset.80eeddbc-1548-45ed-86cd-5e0748e48f08")
)
analysis_only_iver_0_to_6 <- function(weighted_only_iver_0_to_6) {
  
result_df <- create_analysis(weighted_only_iver_0_to_6)
return(result_df)       
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0cbb5b4d-a4c5-488b-8e16-0afd1acbfd32"),
    forest_plot_prep=Input(rid="ri.foundry.main.dataset.54087eba-de49-47d6-a63d-fce8e9409bee")
)
forest_plot <- function(forest_plot_prep) {
strip_data <- forest_plot_prep %>%
  select(id) %>%
  mutate(y_position = 1:nrow(.),
         ymin = y_position - 0.5,
         ymax = y_position + 0.5,
         xmin = -5,
         xmax = 5,
         fill = rep(c("a", "b"), length.out=7)
  ) %>%
  pivot_longer(cols=c(xmin, xmax), names_to="min_max", values_to="x")
  
# CREATE FOREST PLOT
    p_mid <-forest_plot_prep %>%
        ggplot(aes(y = rev(id), group = id)) +
        theme_classic() +
        geom_ribbon(data=strip_data, aes(x=x, ymin=ymin, ymax=ymax, fill=fill, group=id),
              inherit.aes=FALSE, show.legend = FALSE) +
        geom_point(aes(x=exp_estimate), shape=15, size=3) +
        geom_linerange(aes(xmin=exp_ci_low, xmax=exp_ci_high)) +
        labs(x="") +
        coord_cartesian(ylim=c(1,8), xlim=c(0, 3.5))+
        geom_vline(xintercept = 1, linetype="dashed") +
        annotate("text", x = 0.3, y = 8, label = "Met Better") +
        annotate("text", x = 1.75, y = 8, label = "Comp Better") +
          scale_fill_manual(values = c("white","lightgrey")) + 
        theme(axis.line.y = element_blank(),
                axis.ticks.y= element_blank(),
                axis.text.y= element_blank(),
                axis.title.y= element_blank()) 

# add label helpers for plotting
res_plot <- forest_plot_prep %>%
  bind_rows(data.frame(subgroup = "Group", treatment_window = "Treat Window", plot_counts_control = "Comp", plot_counts_metformin = "Met", term2 = "Term", plot_or = "Odds Ratio(95% CI)", pval="p-value")) %>%
  mutate(model = fct_rev(fct_relevel(subgroup, "Group"))) 

# PLOT LEFT SIDE
# left side includes cohort group, term, and weighted outcomes
 p_left <- res_plot  %>%
  ggplot(aes(y = fct_rev(as.factor(id)))) +
  geom_text(aes(x=0, label=subgroup), hjust=0) +
  geom_text(aes(x=1.25, label=treatment_window), hjust=0) +
  #geom_text(aes(x=1, label=term2), hjust=0) +
    geom_text(aes(x=2, label=plot_counts_metformin), hjust=0) +
    geom_text(aes(x=3, label=plot_counts_control), hjust=0) +
 scale_fill_manual(values = c("white", "lightgrey")) + 
  theme_void() +
  coord_cartesian(xlim=c(0,4.35))

# PLOT RIGHT SIDE
## include formated OR and 95%CI and p-value
 p_right <- res_plot %>%
ggplot(aes(y = fct_rev(as.factor(id)))) +
  geom_text(aes(x=0, label=plot_or), hjust=0) +
  geom_text(aes(x=0.5, label=pval), hjust=0) +
  scale_fill_manual(values = c("white", "grey50")) + 
  theme_void() +
    coord_cartesian(xlim=c(0,1))

# layout design
layout <- c(
  area(t = 0, l = 0, b = 8, r = 11.5),
  area(t = 1, l = 11.5, b = 8, r = 15),
  area(t = 0, l = 16, b = 8, r = 20.5)
)
# final plot arrangement
p <- p_left + p_mid + p_right + plot_layout(design = layout) 

print(p)
print(p_mid)
print(p_left)
print(p_right)

return(res_plot)   
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.54087eba-de49-47d6-a63d-fce8e9409bee"),
    forest_prep_counts=Input(rid="ri.foundry.main.dataset.dfafc8b6-0b41-488c-99ab-c6ece3b092b5"),
    forest_prep_glm=Input(rid="ri.foundry.main.dataset.615e02a1-b5ff-4bc4-b6e4-ea725d194e56")
)
forest_plot_prep <- function(forest_prep_counts, forest_prep_glm) {

# MERGE and CLEAN
# merge counts and glm results and select columns of interest
forest_prep_glm$id  <- 1:nrow(forest_prep_glm)
merged_df <- merge(forest_prep_glm, forest_prep_counts, by=c("subgroup","treatment_window")) # NA's match
merged_df <- merged_df[order(merged_df$id), ]
newDF <- merged_df %>%
    select(subgroup, treatment_window, term, plot_or, pval, exp_estimate, exp_ci_low, exp_ci_high, plot_counts_control, n_outcome_censor_control, n_total_censor_control, per_control, plot_counts_metformin, n_outcome_censor_metformin, n_total_censor_metformin, per_metformin)

newDF$id  <- 1:nrow(newDF)

return(newDF)
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.dfafc8b6-0b41-488c-99ab-c6ece3b092b5"),
    weighted_exc_dm_0_to_1=Input(rid="ri.foundry.main.dataset.858b7fe8-5f0c-40fd-87d3-e62647da0751"),
    weighted_exc_dm_0_to_14=Input(rid="ri.foundry.main.dataset.da1d4195-4ba3-4701-ba41-9e0762505dcd"),
    weighted_exc_dm_0_to_6=Input(rid="ri.foundry.main.dataset.de8ad4b6-6fa7-4fda-8c20-5a2128ff96d7"),
    weighted_exc_fluv_0_to_14=Input(rid="ri.foundry.main.dataset.13f25daf-2333-47c3-945b-7f3c9a50dc47"),
    weighted_exc_fluv_0_to_6=Input(rid="ri.foundry.main.dataset.98e56d58-abc1-43d0-8969-6029af8a416c"),
    weighted_exc_iver_0_to_14=Input(rid="ri.foundry.main.dataset.1e41bf28-4dd2-461c-b56f-8229ffec2c54"),
    weighted_exc_iver_0_to_6=Input(rid="ri.foundry.main.dataset.66bd84f9-fbce-497e-954d-256be5d9facb")
)
forest_prep_counts <- function(weighted_exc_fluv_0_to_6, weighted_exc_fluv_0_to_14, weighted_exc_iver_0_to_6, weighted_exc_iver_0_to_14, weighted_exc_dm_0_to_1, weighted_exc_dm_0_to_6, weighted_exc_dm_0_to_14) {

# CLEAN AND CALCUALTE
## calculate the weighted % combined_outcome
## add labels to identify analysis group and to join to later tables
## grab only needed columns
## repeat for all models

m1 <- weighted_exc_dm_0_to_1 %>%
    group_by(cohort) %>%
    count(combined_outcome, wt=weights) %>%
    mutate(n_outcome = n,
        n_outcome_censor = ifelse(n_outcome <20, "<20", paste0(n_outcome)),
            n_total = sum(n),
            n_total_censor = ifelse(n_outcome <20, "<20", paste0(n_total)),
            per = n/sum(n),
            plot_counts = ifelse(n_outcome <20, paste0("<20"," (", format(round(per*100, 1), nsmall=1), "%)"), paste0(round(n), "/", round(sum(n)), " (", format(round(per*100, 1), nsmall=1), "%)")),
        subgroup = 'Excluding DM Indications', 
        treatment_window = '0 to 1') %>%
    select(subgroup, treatment_window, cohort, combined_outcome, n_outcome_censor, n_total_censor, per, plot_counts) %>%
    filter(combined_outcome ==1 )

m2 <- weighted_exc_dm_0_to_6 %>%
    group_by(cohort) %>%
    count(combined_outcome, wt=weights) %>%
    mutate(n_outcome = n,
        n_outcome_censor = ifelse(n_outcome <20, "<20", paste0(n_outcome)),
            n_total = sum(n),
            n_total_censor = ifelse(n_outcome <20, "<20", paste0(n_total)),
            per = n/sum(n),
            plot_counts = ifelse(n_outcome <20, paste0("<20"," (", format(round(per*100, 1), nsmall=1), "%)"), paste0(round(n), "/", round(sum(n)), " (", format(round(per*100, 1), nsmall=1), "%)")),
        subgroup = 'Excluding DM Indications', 
        treatment_window = '0 to 6') %>%
    select(subgroup, treatment_window, cohort, combined_outcome, n_outcome_censor, n_total_censor, per, plot_counts) %>%
    filter(combined_outcome ==1 )

m3 <- weighted_exc_dm_0_to_14 %>%
    group_by(cohort) %>%
    count(combined_outcome, wt=weights) %>%
    mutate(n_outcome = n,
        n_outcome_censor = ifelse(n_outcome <20, "<20", paste0(n_outcome)),
            n_total = sum(n),
            n_total_censor = ifelse(n_outcome <20, "<20", paste0(n_total)),
            per = n/sum(n),
            plot_counts = ifelse(n_outcome <20, paste0("<20"," (", format(round(per*100, 1), nsmall=1), "%)"), paste0(round(n), "/", round(sum(n)), " (", format(round(per*100, 1), nsmall=1), "%)")),
        subgroup = 'Excluding DM Indications', 
        treatment_window = '0 to 14') %>%
    select(subgroup, treatment_window, cohort, combined_outcome, n_outcome_censor, n_total_censor, per, plot_counts) %>%
    filter(combined_outcome ==1 )

m4 <- weighted_exc_fluv_0_to_6 %>%
    group_by(cohort) %>%
    count(combined_outcome, wt=weights) %>%
    mutate(n_outcome = n,
        n_outcome_censor = ifelse(n_outcome <20, "<20", paste0(n_outcome)),
            n_total = sum(n),
            n_total_censor = ifelse(n_outcome <20, "<20", paste0(n_total)),
            per = n/sum(n),
            plot_counts = ifelse(n_outcome <20, paste0("<20"," (", format(round(per*100, 1), nsmall=1), "%)"), paste0(round(n), "/", round(sum(n)), " (", format(round(per*100, 1), nsmall=1), "%)")),
        subgroup = 'Excluding Fluvoxamine', 
        treatment_window = '0 to 6') %>%
    select(subgroup, treatment_window, cohort, combined_outcome, n_outcome_censor, n_total_censor, per, plot_counts) %>%
    filter(combined_outcome ==1 )

m5 <- weighted_exc_fluv_0_to_14 %>%
    group_by(cohort) %>%
    count(combined_outcome, wt=weights) %>%
    mutate(n_outcome = n,
        n_outcome_censor = ifelse(n_outcome <20, "<20", paste0(n_outcome)),
            n_total = sum(n),
            n_total_censor = ifelse(n_outcome <20, "<20", paste0(n_total)),
            per = n/sum(n),
            plot_counts = ifelse(n_outcome <20, paste0("<20"," (", format(round(per*100, 1), nsmall=1), "%)"), paste0(round(n), "/", round(sum(n)), " (", format(round(per*100, 1), nsmall=1), "%)")),
        subgroup = 'Excluding Fluvoxamine', 
        treatment_window = '0 to 14') %>%
    select(subgroup, treatment_window, cohort, combined_outcome, n_outcome_censor, n_total_censor, per, plot_counts) %>%
    filter(combined_outcome ==1 )

m6 <- weighted_exc_iver_0_to_6 %>%
    group_by(cohort) %>%
    count(combined_outcome, wt=weights) %>%
    mutate(n_outcome = n,
        n_outcome_censor = ifelse(n_outcome <20, "<20", paste0(n_outcome)),
            n_total = sum(n),
            n_total_censor = ifelse(n_outcome <20, "<20", paste0(n_total)),
            per = n/sum(n),
            plot_counts = ifelse(n_outcome <20, paste0("<20"," (", format(round(per*100, 1), nsmall=1), "%)"), paste0(round(n), "/", round(sum(n)), " (", format(round(per*100, 1), nsmall=1), "%)")),
        subgroup = 'Excluding Ivermectin', 
        treatment_window = '0 to 6') %>%
    select(subgroup, treatment_window, cohort, combined_outcome, n_outcome_censor, n_total_censor, per, plot_counts) %>%
    filter(combined_outcome ==1 )

m7 <- weighted_exc_iver_0_to_14 %>%
    group_by(cohort) %>%
    count(combined_outcome, wt=weights) %>%
    mutate(n_outcome = n,
        n_outcome_censor = ifelse(n_outcome <20, "<20", paste0(n_outcome)),
            n_total = sum(n),
            n_total_censor = ifelse(n_outcome <20, "<20", paste0(n_total)),
            per = n/sum(n),
            plot_counts = ifelse(n_outcome <20, paste0("<20"," (", format(round(per*100, 1), nsmall=1), "%)"), paste0(round(n), "/", round(sum(n)), " (", format(round(per*100, 1), nsmall=1), "%)")),
        subgroup = 'Excluding Ivermectin', 
        treatment_window = '0 to 14') %>%
    select(subgroup, treatment_window, cohort, combined_outcome, n_outcome_censor, n_total_censor, per, plot_counts) %>%
    filter(combined_outcome ==1 )

# combine models into one dataframe
all_df <- rbind(m1,m2,m3,m4,m5,m6,m7)

# pivot_wider to desired columns for later join and plotting
# deduplicate 
df_f <- all_df %>%
  pivot_wider(
    names_from = cohort,
    values_from = c(plot_counts, n_outcome_censor, n_total_censor, per)
  ) 
#df_g <- df_f %>%
 #   select(subgroup, treatment_window, control, metformin) %>%
  #  group_by(subgroup, treatment_window) %>%
   # mutate(control = dplyr::first(na.omit(control)),
  #  metformin = dplyr::first(na.omit(metformin))
  #  ) %>%
  #  distinct()
    
return(df_f)
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.615e02a1-b5ff-4bc4-b6e4-ea725d194e56"),
    analysis_exc_dm_0_to_1=Input(rid="ri.foundry.main.dataset.927489b6-d598-430f-bfc5-e70cb0d08f77"),
    analysis_exc_dm_0_to_14=Input(rid="ri.foundry.main.dataset.c244a361-028e-486d-a235-b5ce93205ccf"),
    analysis_exc_dm_0_to_6=Input(rid="ri.foundry.main.dataset.5761aa39-71fc-479d-9a85-adb0da0e9a98"),
    analysis_exc_fluv_0_to_14=Input(rid="ri.foundry.main.dataset.a6f4a2ac-4439-479b-bb3a-e26936e8145e"),
    analysis_exc_fluv_0_to_6=Input(rid="ri.foundry.main.dataset.85aa4d90-a3a1-456d-aca7-30e7b788590f"),
    analysis_exc_iver_0_to_14=Input(rid="ri.foundry.main.dataset.c09acbfc-435d-4d4c-9262-2cbb1f00d988"),
    analysis_exc_iver_0_to_6=Input(rid="ri.foundry.main.dataset.a487ee0c-e5f5-4620-bb43-d9856e557761")
)
forest_prep_glm <- function(analysis_exc_fluv_0_to_6, analysis_exc_fluv_0_to_14,analysis_exc_iver_0_to_6, analysis_exc_iver_0_to_14, analysis_exc_dm_0_to_1, analysis_exc_dm_0_to_6, analysis_exc_dm_0_to_14) {

# CLEAN and COMBINE
# want to combine all model results for later plotting and export
# filter to desired (glm) analysis results
# add analysis labels to identify cohort
# grab desired columns

m1 <- analysis_exc_dm_0_to_1 %>%
    filter(analysis == 'glm' & term != '(Intercept)') %>%
    mutate(subgroup = 'Excluding DM Indications', 
            treatment_window = '0 to 1') %>%
    select(subgroup, treatment_window, analysis, term, estimate, p_value, conf_low, conf_high)

m2 <- analysis_exc_dm_0_to_6 %>%
    filter(analysis == 'glm' & term != '(Intercept)') %>%
    mutate(subgroup = 'Excluding DM Indications', 
            treatment_window = '0 to 6') %>%
    select(subgroup, treatment_window, analysis, term, estimate, p_value, conf_low, conf_high)

m3 <- analysis_exc_dm_0_to_14 %>%
    filter(analysis == 'glm' & term != '(Intercept)') %>%
    mutate(subgroup = 'Excluding DM Indications', 
            treatment_window = '0 to 14') %>%
    select(subgroup, treatment_window, analysis, term, estimate, p_value, conf_low, conf_high)

m4 <- analysis_exc_fluv_0_to_6 %>%
    filter(analysis == 'glm' & term != '(Intercept)') %>%
    mutate(subgroup = 'Excluding Fluvoxamine', 
            treatment_window = '0 to 6') %>%
    select(subgroup, treatment_window, analysis, term, estimate, p_value, conf_low, conf_high)

m5 <- analysis_exc_fluv_0_to_14 %>%
    filter(analysis == 'glm' & term != '(Intercept)') %>%
    mutate(subgroup = 'Excluding Fluvoxamine', 
            treatment_window = '0 to 14') %>%
    select(subgroup, treatment_window, analysis, term, estimate, p_value, conf_low, conf_high)

m6 <- analysis_exc_iver_0_to_6 %>%
    filter(analysis == 'glm' & term != '(Intercept)') %>%
    mutate(subgroup = 'Excluding Ivermectin', 
            treatment_window = '0 to 6') %>%
    select(subgroup, treatment_window, analysis, term, estimate, p_value, conf_low, conf_high)

m7 <- analysis_exc_iver_0_to_14 %>%
    filter(analysis == 'glm' & term != '(Intercept)') %>%
    mutate(subgroup = 'Excluding Ivermectin', 
            treatment_window = '0 to 14') %>%
    select(subgroup, treatment_window, analysis, term, estimate, p_value, conf_low, conf_high)

# combine model results
all_df <- rbind(m1,m2,m3,m4,m5,m6,m7)

# exponentiate results to get OR, created formatted output for later plotting
df <- all_df %>%
    mutate(exp_estimate = exp(estimate),
            exp_ci_low = exp(conf_low),
            exp_ci_high = exp(conf_high),
            plot_or = paste0(format(round(exp_estimate, 2), nsmall=2), " (", format(round(exp_ci_low, 2), nsmall=2), "-", format(round(exp_ci_high, 2), nsmall=2), ")"))

    df$pval <- as.numeric(df$p_value)
    df$pval <- case_when(df$pval < 0.001 ~ "<0.001", 
                         TRUE ~ as.character(format(round(df$pval, 3), nsmall=3)))
return(df)  
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.858b7fe8-5f0c-40fd-87d3-e62647da0751"),
    filter_exc_dm_0_to_1=Input(rid="ri.foundry.main.dataset.51c64dd3-e374-41db-840e-e7b5a4d95750")
)
weighted_exc_dm_0_to_1 <- function(filter_exc_dm_0_to_1) {
 
df <- create_weights_for_exc_dm(filter_exc_dm_0_to_1,  "Excluding DM Indications: 0 to 1 Days")
return(df)      
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.da1d4195-4ba3-4701-ba41-9e0762505dcd"),
    filter_exc_dm_0_to_14=Input(rid="ri.foundry.main.dataset.29d7c574-ec44-48c0-9312-e3b0a9782c0b")
)
weighted_exc_dm_0_to_14 <- function(filter_exc_dm_0_to_14) {
 df <- create_weights_for_exc_dm(filter_exc_dm_0_to_14, "Excluding DM Indications: 0 to 14 Days")
return(df)     
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.de8ad4b6-6fa7-4fda-8c20-5a2128ff96d7"),
    filter_exc_dm_0_to_6=Input(rid="ri.foundry.main.dataset.091a4d9c-3f08-412c-b945-6fe6961b6ec0")
)
weighted_exc_dm_0_to_6 <- function(filter_exc_dm_0_to_6) {
 df <- create_weights_for_exc_dm(filter_exc_dm_0_to_6, "Excluding DM Indications: 0 to 6 Days")
return(df)      
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.3a99f2f9-f4f1-452b-b46f-3cade99093ac"),
    filter_exc_dm_7_to_14=Input(rid="ri.foundry.main.dataset.6bd4e0bb-31c2-47eb-a3e9-565784547701")
)
weighted_exc_dm_7_to_14 <- function(filter_exc_dm_7_to_14) {
df <- create_weights_for_exc_dm(filter_exc_dm_7_to_14, "Excluding DM Indications: 7 to 14 Days")
return(df)  
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.13f25daf-2333-47c3-945b-7f3c9a50dc47"),
    filter_exc_fluv_0_to_14=Input(rid="ri.foundry.main.dataset.fd9b6cd0-6c48-4d41-a321-2c42885abbac")
)
weighted_exc_fluv_0_to_14 <- function(filter_exc_fluv_0_to_14) {
  
df <- create_weights_for_exc_dm(filter_exc_fluv_0_to_14, "Excluding Fluvoxamine: 0 to 14 Days")
return(df)         
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.98e56d58-abc1-43d0-8969-6029af8a416c"),
    filter_exc_fluv_0_to_6=Input(rid="ri.foundry.main.dataset.50bb7017-b91b-489c-9bcd-bc62e7e083b0")
)
weighted_exc_fluv_0_to_6 <- function(filter_exc_fluv_0_to_6) {
  
df <- create_weights_for_exc_dm(filter_exc_fluv_0_to_6, "Excluding Fluvoxamine: 0 to 6 Days")
return(df)  
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.1e41bf28-4dd2-461c-b56f-8229ffec2c54"),
    filter_exc_iver_0_to_14=Input(rid="ri.foundry.main.dataset.2a7faabb-a7f4-4fbf-b40d-bf820b99ee86")
)
weighted_exc_iver_0_to_14 <- function(filter_exc_iver_0_to_14) {
  
df <- create_weights_for_exc_dm(filter_exc_iver_0_to_14,  "Excluding Ivermectin: 0 to 14 Days")
return(df)      
       
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.66bd84f9-fbce-497e-954d-256be5d9facb"),
    filter_exc_iver_0_to_6=Input(rid="ri.foundry.main.dataset.0faf28b4-1e28-40c8-a8cd-bcd880839595")
)
weighted_exc_iver_0_to_6 <- function(filter_exc_iver_0_to_6) {
  
df <- create_weights_for_exc_dm(filter_exc_iver_0_to_6, "Excluding Ivermectin: 0 to 6 Days")
return(df)      
   
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.fbfb30e9-b6ce-45a7-9454-61c539ee4d6b"),
    filter_only_fluv_0_to_14=Input(rid="ri.foundry.main.dataset.d0b4fc62-cc2a-4da4-bb76-6a2e4f84c780")
)
weighted_only_fluv_0_to_14 <- function(filter_only_fluv_0_to_14) {
  
df <- create_weights_for_exc_dm(filter_only_fluv_0_to_14, "Only Fluvoxamine: 0 to 14 Days")
return(df)     
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.80fe94fc-0d6a-4d95-9cdb-582f67dab25f"),
    filter_only_fluv_0_to_6=Input(rid="ri.foundry.main.dataset.8d080864-59f8-4a6a-914e-9845d1532104")
)
weighted_only_fluv_0_to_6 <- function(filter_only_fluv_0_to_6) {
  
df <- create_weights_for_exc_dm(filter_only_fluv_0_to_6,"Only Fluvoxamine: 0 to 6 Days")
return(df)      
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.888bf6bb-87bb-411f-bcbf-c38a2ee90e16"),
    filter_only_iver_0_to_14=Input(rid="ri.foundry.main.dataset.682ab11e-6749-49a8-8266-2d1b0042292c")
)
weighted_only_iver_0_to_14 <- function(filter_only_iver_0_to_14) {
  
df <- create_weights_for_exc_(filter_only_iver_0_to_14,  "Only Ivermectin: 0 to 14 Days")
return(df)      
           
}

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.80eeddbc-1548-45ed-86cd-5e0748e48f08"),
    filter_only_iver_0_to_6=Input(rid="ri.foundry.main.dataset.2d00952b-a922-4fc8-b5c3-61e63e07abd4")
)
weighted_only_iver_0_to_6 <- function(filter_only_iver_0_to_6) {
  
df <- create_weights_for_exc_dm(filter_only_iver_0_to_6,  "Only Ivermectin: 0 to 6 Days")
return(df)      
           
}

