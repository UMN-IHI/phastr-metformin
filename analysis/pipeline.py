import pandas as pd
import numpy as np
import pyspark.sql.functions as F

inclusion_bin_cols = [
    "metformin_before_365",
    "metformin_before_180",
    "metformin_after",
    "ivermectin_after",
    "fluvoxamine_after",
    "montelukast_after",
    "albuterol_after",
    "fluticasone_after",
    "prednisone_after",
    "molnupiravir_after",
    "excl_hosp_event",
    "comparator_meds_all_before_365",
    "comparator_meds_all_after",
    "excl_ckd45",
]

variables_bin_cols = [
    "deceased",
    "ace_before",
    "arb_before",
    "statin_before",
    "anticoagulant_before",
    "aspirin_before",
    "cad",
    "cancer",
    "ckd123",
    "heart_failure",
    "hypertension",
    "liver_disease",
    "t1dm_after",
    "t2dm_after",
    "prediabetes_after",
    "dm_complicated_after",
    "dm_uncomplicated_after",
    "pasc",
    "hosp_outcome",
    "nirmatrelvir",
    "ritonavir",
    "paxlovid",
    "remdesivir",
    "dexamethasone",
    "a1c_gt65_before",
    "nafld_before",
    "adjustment_disorder_before",
    "dysthymia_before",
    "seasonal_affective_disorder_before",
    "hypomania_before",
    "dysphoria_before",
    "insomnia_before",
    "bipolar_before",
    "mood_disorder_before",
    "steroids_before",
    "binge_eating_disorder_before",
    "bulimia_nervosa_before",
    "generalized_anxiety_before",
    "major_depression_disorder_before",
    "ocd_before",
    "panic_disorder_before",
    "ptsd_before",
    "social_anxiety_disorder_before",
    "asthma_before",
    "copd_before",
    "bronchiolitis_obliterans_before",
    "eosinophilic_esophagitis_before",
    "phct_before",
    "mast_cell_activation_before",
    "exercise_induced_asthma_before",
    "exercise_induced_bronchoconstriction_before",
    "allergic_rhinitis_before",
    "prediabetes_before",
    "t1dm_before",
    "t2dm_before",
    "gestational_dm_before",
    "antipsychotic_induced_weight_gain_before",
    "pcos_before",
    "ckd4_before",
    "esrd_dialysis_before",
    "biguanides_before_365",
    "dpp4_before_365",
    "glp1_before_365",
    "sglt2_before_365",
    "su_before_365",
    "thiazo_before_365",
    "insulin_before_365",
    "other_diabetes_med_before_365",
]

new_names = {
    "0_N": "N",
    "age": "Age",
    "Harmonized_age_group":  "Age",
    "Harmonized_gender":  "Gender ",
    "race_ethnicity":  "Race, Ethnicity ",
    "office_visits_0_6": "Office Visits, 0-6 months before",
    "office_visits_6_12": "Office Visits, 6-12 months before",
    "office_visits_after": "Office Visits, after",
    "weight": "Weight",
    "height":  "Height",
    "bmi":  "BMI",
    "bmi_category": "BMI Category",
    "ace_before":  "ACEi",
    "arb_before": "ARB",
    "statin_before":  "Statins ",
    "anticoagulant_before": "Anticoagulant",
    "aspirin_before":  "Aspirin",
    "steroids_before":  "Steroids", 
    "biguanides_before_365":  "Biguanides",
    "dpp4_before_365":  "DPP4i ",
    "glp1_before_365":  "GLP-1 RA",
    "sglt2_before_365":  "SGLT-2 inhibitor",
    "su_before_365":  "Sulfonylureas",
    "thiazo_before_365":  "Thiazolidinediones",
    "insulin_before_365":  "Outpatient Insulin",
    "other_diabetes_med_before_365":  "Other Diabetes meds",
    "cad":  "CAD",
    "cancer":  "Cancer",
    "ckd123":  "CKD, Stage 1-3",
    "heart_failure":  "Heart Failure",
    "hypertension":  "Hypertension",
    "liver_disease":  "Liver Disease",
    "nafld_before":  "NAFLD",
    "adjustment_disorder_before":  "Adjustment d/o",
    "dysthymia_before":  "Dysthymia",
    "seasonal_affective_disorder_before":  "SAD",
    "hypomania_before":  "Hypomania",
    "dysphoria_before":  "Dysphoria",
    "insomnia_before":  "Insomnia ",
    "bipolar_before":  "Bipolar Disease ",
    "mood_disorder_before":  "Mood Disorder",
    "binge_eating_disorder_before":  "Binge eating d/o",
    "bulimia_nervosa_before":  "Bulimia",
    "generalized_anxiety_before":  "GAD",
    "major_depression_disorder_before":  "Depression",
    "ocd_before":  "OCD",
    "panic_disorder_before":  "Panic d/o ",
    "ptsd_before":  "Post Traumatic Stress d/o ",
    "social_anxiety_disorder_before":  "Social Anxiety d/o",
    "asthma_before":  "Asthma",
    "copd_before":  "COPD",
    "bronchiolitis_obliterans_before":  "Bronchiolitis obliterans",
    "eosinophilic_esophagitis_before":  "Eosinophilic esophagitis",
    "phct_before":  "Post-hematopoietic cell transplantation",
    "mast_cell_activation_before":  "Mast Cell Activation",
    "exercise_induced_asthma_before":  "Exercised-induced Asthma",
    "exercise_induced_bronchoconstriction_before":  "Exercised-induced bronchoconstriction",
    "allergic_rhinitis_before":  "Allergic rhinitis",
    "prediabetes_before":  "Prediabetes",
    "t1dm_before":  "Type 1 DM ",
    "t2dm_before":  "Type 2 DM ",
    "gestational_dm_before":  "Gestational DM ",
    "antipsychotic_induced_weight_gain_before":  "Antipsychotic weight gain",
    "pcos_before":  "Polycystic Ovarian Syndrome",
    "a1c_gt65_before":  "HbA1c>=6.5",
    "a1c_last":  "Last HbA1c ",
    "creatinine_before_mean":  "Creatinine",
    "egfr":  "estimated GFR",
    "data_partner_id":  "Data Partner",
    "covid_era": "COVID Era",
    "fluticasone_after": "Fluticasone, after",
    "fluvoxamine_after": "Fluvoxamine, after",
    "montelukast_after": "Montelukast, after",
    "ivermectin_after": "Ivermectin, after",
    "comparator_meds_all_after": "Comparator Meds, all, after",
    "metformin_after": "Metformin, after",
    "molnupiravir_after": "Molnupiravir, after",
    "ritonavir": "Ritonavir, after",
    "nirmatrelvir": "Nirmatrelvir, after",
    "dexamethasone": "Dexamethasone, after",
    "paxlovid": "Paxlovid, after",
    "remdesivir": "Remdesivir, after",
    "control_days_from_index": "Control Days from Index",
    "metformin_days_from_index": "Metformin Days from Index",
    "control_days_from_index_category": "Control Days from Index",
    "metformin_days_from_index_category": "Metformin Days from Index",
    "death_180": "Deceased",
    "outcome": "PASC (any) before 180d"
    }

row_numbers = {}
for index, (key, value) in enumerate(new_names.items()):
    row_numbers[key] = index

from scipy import stats

def create_table1(data,cohort_names,cols,censor=False,row_names=None):
    t1_cols = ['var_type','var_name','condition','c1_mean','c2_mean','c1_std','c2_std','c1_count','c2_count','c1_percent','c2_percent',
            'sdiff', 'test_stat', 'pval', 'text_str1', 'text_str2']
    def t1_cont(c1, c2, var_name):
        c1_n = c1[var_name].apply(pd.to_numeric, errors='coerce').dropna()
        c2_n = c2[var_name].apply(pd.to_numeric, errors='coerce').dropna()
        c1_mean = c1_n.mean()
        c2_mean = c2_n.mean()
        c1_std = c1_n.std()
        c2_std = c2_n.std()
        n1 = len(c1_n)
        n2 = len(c2_n)
        pooled_sd = np.sqrt((c1_std**2 + c2_std**2) / 2)
        if pooled_sd == 0:
            sdiff = 0
        else:
            sdiff = abs((c1_mean - c2_mean) / pooled_sd)
        test_stat, pval = stats.ttest_ind(c1_n, c2_n, equal_var=False)
        text_str1 = f"{c1_mean:>10,.2f} ({c1_std:.2f})"
        text_str2 = f"{c2_mean:>10,.2f} ({c2_std:.2f})"
        return ['cont', var_name, None, c1_mean, c2_mean, c1_std, c2_std,  None, None, None, None,
                sdiff, test_stat, pval, text_str1, text_str2]

    def calc_smd(c1, c2, c1_count, c2_count, var_name, censor_flag=False):
        n1 = len(c1)
        n2 = len(c2)
        c1_percent = c1_count / n1
        c2_percent = c2_count / n2
        pooled_sd = np.sqrt((c1_percent*(1-c1_percent)+c2_percent*(1-c2_percent))/2)
        if pooled_sd == 0:
            sdiff = 0
            test_stat = 0
            pval = 0
        else:
            sdiff = abs((c1_percent - c2_percent) / pooled_sd)
            obs = [[c1_count, (n1-c1_count)],[c2_count,(n2-c2_count)]]
            test_stat, pval, dof, expected = stats.chi2_contingency(obs)
        if censor_flag:
            if (c1_count != 0) and (c1_count < 20):
                text_str1 = f"             < 20"
            else:
                text_str1 = f"{c1_count:>10,d} ({c1_percent:.2f})"
            if (c2_count != 0) and (c2_count < 20):
                text_str2 = f"             < 20"
            else:
                text_str2 = f"{c2_count:>10,d} ({c2_percent:.2f})"
        else:
            text_str1 = f"{c1_count:>10,d} ({c1_percent:.2f})"
            text_str2 = f"{c2_count:>10,d} ({c2_percent:.2f})"
        
        return c1_percent, c2_percent, sdiff, pval, test_stat, text_str1, text_str2

    # For Binary variables, we will compute a count, percent/proportion, standardized difference and pvalue
    def t1_bin(c1, c2, var_name, condition):
        if type(condition) == int:
            cond_st = f"{condition}"        
        else:
            cond_st = f"'{condition}'"
        c1_cond =  c1.query(f"{var_name} == {cond_st}")[var_name]
        c2_cond = c2.query(f"{var_name} == {cond_st}")[var_name]
        c1_count = c1_cond.count()
        c2_count = c2_cond.count()
        c1_percent, c2_percent, sdiff, pval, test_stat, text_str1, text_str2 = calc_smd(c1, c2, c1_count, c2_count, var_name, censor_flag=censor)
        return ['bin', var_name, condition, None, None, None, None, c1_count, c2_count, c1_percent, c2_percent, 
                sdiff, test_stat, pval, text_str1, text_str2]

    # For Categorical variables, we will treat each category as it's own binary variable 
    def t1_cat(c1, c2, var_name, levels):
        results = {}
        for l in levels:
            z = t1_bin(c1, c2, var_name, l)
            results[l] = z
        return results

    def display_t1_item(t1_item):
        df = pd.DataFrame(t1_item).T
        df.columns = columns=t1_cols
        display(df)

    # Start of create_table1 function

    cohort = data.toPandas()
    cohorts = {}
    for c in cohort_names:
        cohorts[c] = cohort.query(f"cohort == '{c}'")

    c1 = cohorts[cohort_names[0]]
    c2 = cohorts[cohort_names[1]]

    table1 = pd.DataFrame(columns=['var_text'] + t1_cols)

    # Insert N
    N = len(c1) + len(c2)
    c1_count = len(c1)
    c2_count = len(c2)
    c1_percent, c2_percent, sdiff, pval, test_stat, text_str1, text_str2 = calc_smd(c1, c2, c1_count, c2_count, "_N")

    table1.loc[0] = ["0_N","cont", "_N", 1, None, None, None, None,  c1_count, c2_count, c1_percent, c2_percent, 
                sdiff, pval, test_stat, text_str1, text_str2]

    # Binary variables
    for kind in ["bin","cont","cat"]:
        for v in cols[kind]:
            var_name = v
            k = v   
            if kind == 'cont':
                res = t1_cont(c1, c2, var_name)
                row = [k] + res
                table1.loc[len(table1)] = row
            elif kind == 'bin':
                condition = 1
                res = t1_bin(c1, c2, var_name, condition)
                row = [k] + res
                table1.loc[len(table1)] = row
            elif kind == 'cat':
                levels = list(cohort.groupby([var_name]).groups.keys())
                res = t1_cat(c1, c2, var_name, levels)
                # Unpack each row of the categorical results
                for lvl, r in res.items():
                    l1 = f"{k}|{lvl}"
                    row = [l1] + r
                    table1.loc[len(table1)] = row
            else:
                raise InputException(f"Invalid variable type: {func}")
            
    # Print out Table 1

    template = "{:<70}  {:>20}   {:>20}   {:>10}"
    print(template.format('characteristic',cohort_names[0],cohort_names[1],'Standard Difference'))
    print(template.format('',"n = {:,d}".format(len(c1)),"n = {:,d}".format(len(c2)),''))
    print(template.format('=========================================================','==============','==============','==================='))

    for i, row in table1.iterrows():
        str = template.format(row.var_text, row.text_str1, row.text_str2, f"{row.sdiff:.3f}")
        print(str)

    # Create a dataframe for Table 1
    df_table1 = pd.DataFrame(columns=['item_num','varname','characteristic',cohort_names[0],cohort_names[1],'SMD'])
    for i, row in table1.iterrows():
        row_i = f"{i:0>5}"
        varname_full = row.var_text
        if "|" in varname_full:
            parts = varname_full.split("|")
            varname = parts[0]
            var_rest = parts[1]
        else:
            varname = varname_full
            var_rest = ""
        if row_names == None:
            df_table1.loc[i] = [row_i, varname, varname, row.text_str1, row.text_str2, f"{row.sdiff:.3f}"]
        else:
            if varname in row_names:
                desc = row_names[varname]
                if var_rest != "":
                    desc = f"{desc}|{var_rest}"
                    row_i = f"{row_numbers[varname]:0>5}.{var_rest[0:2]}"
                else:
                    row_i = f"{row_numbers[varname]:0>5}"
            else:
                desc = varname
            df_table1.loc[i] = [row_i, row.var_text, desc, row.text_str1, row.text_str2, f"{row.sdiff:.3f}"]

    sdf = spark.createDataFrame(df_table1)
    return sdf

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.51c64dd3-e374-41db-840e-e7b5a4d95750"),
    fact_table_final=Input(rid="ri.foundry.main.dataset.78944dcb-af75-49bc-b8e3-1f041c3e6c25")
)
def filter_exc_dm_0_to_1(fact_table_final):
    exclusions = ["prediabetes_before", "t1dm_before", "t2dm_before", "gestational_dm_before", "antipsychotic_induced_weight_gain_before", "pcos_before"]
    sql = " and ".join([f"(coalesce({x},0) = 0)" for x in exclusions])
    print(sql)
    df = fact_table_final.filter(F.expr(sql))

    df = df.filter(F.expr("(combined_days_from_index >= 0) and (combined_days_from_index <= 1)"))

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.29d7c574-ec44-48c0-9312-e3b0a9782c0b"),
    fact_table_final=Input(rid="ri.foundry.main.dataset.78944dcb-af75-49bc-b8e3-1f041c3e6c25")
)
def filter_exc_dm_0_to_14(fact_table_final):
    exclusions = ["prediabetes_before", "t1dm_before", "t2dm_before", "gestational_dm_before", "antipsychotic_induced_weight_gain_before", "pcos_before"]
    sql = " and ".join([f"(coalesce({x},0) = 0)" for x in exclusions])
    print(sql)
    df = fact_table_final.filter(F.expr(sql))

    df = df.filter(F.expr("(combined_days_from_index >= 0) and (combined_days_from_index <= 14)"))

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.091a4d9c-3f08-412c-b945-6fe6961b6ec0"),
    fact_table_final=Input(rid="ri.foundry.main.dataset.78944dcb-af75-49bc-b8e3-1f041c3e6c25")
)
def filter_exc_dm_0_to_6(fact_table_final):
    exclusions = ["prediabetes_before", "t1dm_before", "t2dm_before", "gestational_dm_before", "antipsychotic_induced_weight_gain_before", "pcos_before"]
    sql = " and ".join([f"(coalesce({x},0) = 0)" for x in exclusions])
    print(sql)
    df = fact_table_final.filter(F.expr(sql))

    df = df.filter(F.expr("(combined_days_from_index >= 0) and (combined_days_from_index <= 6)"))

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.6bd4e0bb-31c2-47eb-a3e9-565784547701"),
    fact_table_final=Input(rid="ri.foundry.main.dataset.78944dcb-af75-49bc-b8e3-1f041c3e6c25")
)
def filter_exc_dm_7_to_14(fact_table_final):
    exclusions = ["prediabetes_before", "t1dm_before", "t2dm_before", "gestational_dm_before", "antipsychotic_induced_weight_gain_before", "pcos_before"]
    sql = " and ".join([f"(coalesce({x},0) = 0)" for x in exclusions])
    print(sql)
    df = fact_table_final.filter(F.expr(sql))
    
    df = df.filter(F.expr("(combined_days_from_index >= 7) and (combined_days_from_index <= 14)"))

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.fd9b6cd0-6c48-4d41-a321-2c42885abbac"),
    fact_table_final=Input(rid="ri.foundry.main.dataset.78944dcb-af75-49bc-b8e3-1f041c3e6c25")
)
def filter_exc_fluv_0_to_14(fact_table_final):
    exclusions = ["prediabetes_before", "t1dm_before", "t2dm_before", "gestational_dm_before", "antipsychotic_induced_weight_gain_before", "pcos_before"]
    sql = " and ".join([f"(coalesce({x},0) = 0)" for x in exclusions])
    print(sql)
    df = fact_table_final.filter(F.expr(sql))
    df = df.filter(F.expr("((combined_days_from_index >= 0) and (combined_days_from_index <= 14)) and (cohort = 'metformin' or (cohort = 'control' and fluvoxamine_after = 0))"))

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.50bb7017-b91b-489c-9bcd-bc62e7e083b0"),
    fact_table_final=Input(rid="ri.foundry.main.dataset.78944dcb-af75-49bc-b8e3-1f041c3e6c25")
)
def filter_exc_fluv_0_to_6(fact_table_final):
    exclusions = ["prediabetes_before", "t1dm_before", "t2dm_before", "gestational_dm_before", "antipsychotic_induced_weight_gain_before", "pcos_before"]
    sql = " and ".join([f"(coalesce({x},0) = 0)" for x in exclusions])
    print(sql)
    df = fact_table_final.filter(F.expr(sql))   
    df = df.filter(F.expr("(combined_days_from_index >= 0) and (combined_days_from_index <= 6) and (cohort = 'metformin' or (cohort = 'control' and fluvoxamine_after = 0))"))

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2a7faabb-a7f4-4fbf-b40d-bf820b99ee86"),
    fact_table_final=Input(rid="ri.foundry.main.dataset.78944dcb-af75-49bc-b8e3-1f041c3e6c25")
)
def filter_exc_iver_0_to_14(fact_table_final):
    exclusions = ["prediabetes_before", "t1dm_before", "t2dm_before", "gestational_dm_before", "antipsychotic_induced_weight_gain_before", "pcos_before"]
    sql = " and ".join([f"(coalesce({x},0) = 0)" for x in exclusions])
    print(sql)
    df = fact_table_final.filter(F.expr(sql))
    df = df.filter(F.expr("((combined_days_from_index >= 0) and (combined_days_from_index <= 14)) and (cohort = 'metformin' or (cohort = 'control' and ivermectin_after = 0))"))

    return df      

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0faf28b4-1e28-40c8-a8cd-bcd880839595"),
    fact_table_final=Input(rid="ri.foundry.main.dataset.78944dcb-af75-49bc-b8e3-1f041c3e6c25")
)
def filter_exc_iver_0_to_6(fact_table_final):
    exclusions = ["prediabetes_before", "t1dm_before", "t2dm_before", "gestational_dm_before", "antipsychotic_induced_weight_gain_before", "pcos_before"]
    sql = " and ".join([f"(coalesce({x},0) = 0)" for x in exclusions])
    print(sql)
    df = fact_table_final.filter(F.expr(sql))
    df = df.filter(F.expr("((combined_days_from_index >= 0) and (combined_days_from_index <= 6)) and (cohort = 'metformin' or (cohort = 'control' and ivermectin_after = 0))"))

    return df    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d0b4fc62-cc2a-4da4-bb76-6a2e4f84c780"),
    fact_table_final=Input(rid="ri.foundry.main.dataset.78944dcb-af75-49bc-b8e3-1f041c3e6c25")
)
def filter_only_fluv_0_to_14(fact_table_final):
    exclusions = ["prediabetes_before", "t1dm_before", "t2dm_before", "gestational_dm_before", "antipsychotic_induced_weight_gain_before", "pcos_before"]
    sql = " and ".join([f"(coalesce({x},0) = 0)" for x in exclusions])
    print(sql)
    df = fact_table_final.filter(F.expr(sql))
    df = df.filter(F.expr("((combined_days_from_index >= 0) and (combined_days_from_index <= 14)) and (cohort = 'metformin' or (cohort = 'control' and fluvoxamine_after = 1 and fluticasone_after = 0 and montelukast_after = 0 and ivermectin_after = 0))"))

    return df     

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.8d080864-59f8-4a6a-914e-9845d1532104"),
    fact_table_final=Input(rid="ri.foundry.main.dataset.78944dcb-af75-49bc-b8e3-1f041c3e6c25")
)
def filter_only_fluv_0_to_6(fact_table_final):
    exclusions = ["prediabetes_before", "t1dm_before", "t2dm_before", "gestational_dm_before", "antipsychotic_induced_weight_gain_before", "pcos_before"]
    sql = " and ".join([f"(coalesce({x},0) = 0)" for x in exclusions])
    print(sql)
    df = fact_table_final.filter(F.expr(sql))
    df = df.filter(F.expr("((combined_days_from_index >= 0) and (combined_days_from_index <= 6)) and (cohort = 'metformin' or (cohort = 'control' and fluvoxamine_after = 1 and fluticasone_after = 0 and montelukast_after = 0 and ivermectin_after = 0))"))

    return df    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.682ab11e-6749-49a8-8266-2d1b0042292c"),
    fact_table_final=Input(rid="ri.foundry.main.dataset.78944dcb-af75-49bc-b8e3-1f041c3e6c25")
)
def filter_only_iver_0_to_14(fact_table_final):
    exclusions = ["prediabetes_before", "t1dm_before", "t2dm_before", "gestational_dm_before", "antipsychotic_induced_weight_gain_before", "pcos_before"]
    sql = " and ".join([f"(coalesce({x},0) = 0)" for x in exclusions])
    print(sql)
    df = fact_table_final.filter(F.expr(sql)) 
    df = df.filter(F.expr("((combined_days_from_index >= 0) and (combined_days_from_index <= 14)) and (cohort = 'metformin' or (cohort = 'control' and fluvoxamine_after = 0 and fluticasone_after = 0 and montelukast_after = 0 and ivermectin_after = 1))"))

    return df         

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2d00952b-a922-4fc8-b5c3-61e63e07abd4"),
    fact_table_final=Input(rid="ri.foundry.main.dataset.78944dcb-af75-49bc-b8e3-1f041c3e6c25")
)
def filter_only_iver_0_to_6(fact_table_final):
    exclusions = ["prediabetes_before", "t1dm_before", "t2dm_before", "gestational_dm_before", "antipsychotic_induced_weight_gain_before", "pcos_before"]
    sql = " and ".join([f"(coalesce({x},0) = 0)" for x in exclusions])
    print(sql)
    df = fact_table_final.filter(F.expr(sql))
    df = df.filter(F.expr("((combined_days_from_index >= 0) and (combined_days_from_index <= 6)) and (cohort = 'metformin' or (cohort = 'control' and fluvoxamine_after = 0 and fluticasone_after = 0 and montelukast_after = 0 and ivermectin_after = 1))"))

    return df     

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d3691011-96e7-40dc-bfa9-b6c26043177b"),
    filter_exc_dm_0_to_1=Input(rid="ri.foundry.main.dataset.51c64dd3-e374-41db-840e-e7b5a4d95750"),
    filter_exc_dm_0_to_14=Input(rid="ri.foundry.main.dataset.29d7c574-ec44-48c0-9312-e3b0a9782c0b"),
    filter_exc_dm_0_to_6=Input(rid="ri.foundry.main.dataset.091a4d9c-3f08-412c-b945-6fe6961b6ec0"),
    filter_exc_fluv_0_to_14=Input(rid="ri.foundry.main.dataset.fd9b6cd0-6c48-4d41-a321-2c42885abbac"),
    filter_exc_fluv_0_to_6=Input(rid="ri.foundry.main.dataset.50bb7017-b91b-489c-9bcd-bc62e7e083b0"),
    filter_exc_iver_0_to_14=Input(rid="ri.foundry.main.dataset.2a7faabb-a7f4-4fbf-b40d-bf820b99ee86"),
    filter_exc_iver_0_to_6=Input(rid="ri.foundry.main.dataset.0faf28b4-1e28-40c8-a8cd-bcd880839595")
)
def table1_all_censored(filter_exc_fluv_0_to_6, filter_exc_fluv_0_to_14,filter_exc_iver_0_to_6, filter_exc_iver_0_to_14,  filter_exc_dm_0_to_1, filter_exc_dm_0_to_6, filter_exc_dm_0_to_14):

    fact_tables = {
      "filter_exc_fluv_0_to_6": filter_exc_fluv_0_to_6,
      "filter_exc_fluv_0_to_14": filter_exc_fluv_0_to_14,
     "filter_exc_iver_0_to_6": filter_exc_iver_0_to_6,
      "filter_exc_iver_0_to_14": filter_exc_iver_0_to_14, 
      "filter_exc_dm_0_to_1": filter_exc_dm_0_to_1, 
      "filter_exc_dm_0_to_6": filter_exc_dm_0_to_6, 
      "filter_exc_dm_0_to_14": filter_exc_dm_0_to_14
    }
    cohort_names = ['metformin','control']
    cols = {}
    new_bin_cols = ['outcome','death_180'] 
    new_cont_cols = [f"{v}_days_from_index" for v in cohort_names] + ['age','egfr'] 
    new_cat_cols = [f"{v}_days_from_index_category" for v in cohort_names] 
    remove_cols_bin = ["ckd4_before", "esrd_dialysis_before", "excl_ckd45", "excl_hosp_event", "hosp_outcome", "mast_cell_activation_before", "metformin_before_180", "metformin_before_365", "prediabetes_after", "prednisone_after", "albuterol_after", "t1dm_after", "t2dm_after", "dm_complicated_after", "dm_uncomplicated_after", "comparator_meds_all_before_365","pasc","deceased","biguanides_before_365"]
    remove_cols_cont = ["bmi_computed", "bmi_site","height","weight"]
    c = inclusion_bin_cols + variables_bin_cols + new_bin_cols
    for i in remove_cols_bin:
        if i in c:
            c.remove(i)
    cols["bin"] = c
    cols["cont"] = ['office_visits_0_6', 'office_visits_6_12', 'office_visits_after', 
                    'bmi', 'a1c_last', 'creatinine_before_mean'] + new_cont_cols

    cols["cat"] = ['Harmonized_gender','Harmonized_age_group', 'race_ethnicity','covid_era', 'bmi_category'] + new_cat_cols
    
    tables = []
    for name, ft in fact_tables.items():
        print("Name = ", name)
        sdf = create_table1(ft, cohort_names, cols, censor=True, row_names=new_names)
        sdf = sdf.withColumn("table1_name",F.lit(name))
        sdf = sdf.select(["table1_name","item_num","varname","characteristic","metformin","control","SMD"])
        sdf.show()
        tables.append(sdf)

    from functools import reduce
    from pyspark.sql import DataFrame
    all = reduce(DataFrame.union, tables)

    return all
