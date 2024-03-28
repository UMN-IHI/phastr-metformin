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

def make_vars(slice_num,cohort, person, condition_occurrence, drug_exposure, measurement,visit_occurrence, procedure_occurrence, observation, concept_set_members, death, concept,slice_width=25,var_defs=None):

    left_join_cte = cohort
    rets = {}

    start_index = slice_num * slice_width
    end_index = start_index + slice_width
    var_list = []
    i = 0
    defs_slice = dict(list(var_defs.items())[start_index:end_index])
    for var_name, row in defs_slice.items():
        print(i, ": ", var_name, ": ")
        var_full_name = f"{var_name}"

        st = row + f"\nrets['{var_full_name}']={var_full_name}_cte"
        print(st)
        exec(st)
        cte = rets[f'{var_full_name}']

        cte = cte.withColumnRenamed("var_val",f"{var_full_name}")
        cte = cte.withColumnRenamed("var_date",f"{var_full_name}_date")
        left_join_cte = left_join_cte.join(cte.alias(var_full_name), "cohort_person_id", how='left')
        var_list += [var_full_name]
        i = i + 1

    return left_join_cte

def my_codesets_icd(concept_set_members, concept, concept_relationship, concept_ancestor):

    # Compute Condition concept sets from ICD10
    conditions = {
        "binge_eating_disorder": ['F50.8'],
        "bulimia_nervosa": ['F50.2'],
        "major_depression_disorder": ['F32', 'F33'],
        "panic_disorder": ['F41.0'],
        "ptsd": ['F43.10'],
        "social_anxiety_disorder": ['F40.10'],
        "bronchiolitis_obliterans": ['J44.81', 'J84.89'],
        "eosinophilic_esophagitis": ['K20.0'],
        "phct": ['M31.11'],
        "mast_cell_activation": ['F32', 'F33'],
        "exercise_induced_asthma": ['J45.990'],
        "exercise_induced_bronchoconstriction": ['J45.990'],
        "allergic_rhinitis": ['J30.9'],
        "antipsychotic_induced_weight_gain": ['R63.5', 'T50.905A', 'R63.5', 'T43.505'],
        "adjustment_disorder": ['F43.2'],
        "dysthymia": ['F34.1'],
        "seasonal_affective_disorder": ['F33.9'],
        "hypomania": ['F31'],
        "dysphoria": ['F64.9'],
        "generalized_anxiety": ['F41.1'],
        "insomnia": ['G47'],
        "bipolar": ['F06.33', 'F06.34', 'F30.10', 'F30.11', 'F30.12', 'F30.13', 'F30.2', 'F30.3', 'F30.8', 'F30.9', 'F31.0', 'F31.10', 'F31.11', 'F31.12', 'F31.13', 'F31.2', 'F31.30', 'F31.31', 'F31.32', 'F31.4', 'F3.15', 'F31.60', 'F31.61', 'F31.62', 'F31.63', 'F31.64', 'F31.71', 'F31.73', 'F31.75', 'F31.77', 'F31.81', 'F31.89', 'F31.9', 'F34.0'],
        "mood_disorder": ['F34.8', 'F34.81', 'F34.89', 'F34.9', 'F39'],
    }

    df = pd.concat({k: pd.Series(v) for k, v in conditions.items()})
    df = df.reset_index()
    df.columns = ['codeset_id','level','icd10_code']
    icd10 = spark.createDataFrame(df[['codeset_id','icd10_code']])
 
    # Find concept_ids for the icd10 codes
    icd10_concept_ids = icd10.join(concept,(icd10.icd10_code == concept.concept_code) & (concept.vocabulary_id == 'ICD10CM'))

    cols = ['codeset_id', 'concept_id', 'concept_code','concept_name', 'vocabulary_id']
    #other_cols = ["concept_code","vocabulary_id","standard_concept"]
    other_cols = []

    #icd10_concept_ids = icd10_concept_ids[cols]
    icd10_concept_ids = icd10_concept_ids.selectExpr("codeset_id", "concept_id", "codeset_id as concept_set_name", "cast(1 as boolean) as is_most_recent_version", "1 as version", "concept_name", "cast(1 as boolean) as archived", *other_cols)
    icd10_concept_ids.show()

    dfs = icd10_concept_ids.selectExpr("codeset_id","concept_id as icd10_id").join(concept_relationship,F.expr("icd10_id = concept_id_1 and relationship_id = 'Maps to'"),how="inner")
    dfs = dfs.join(concept.alias("c1"),F.expr("concept_id_2 = c1.concept_id"))
    dfs.show()
    df2 = dfs.selectExpr("codeset_id", "concept_id", "codeset_id as concept_set_name", "cast(1 as boolean) as is_most_recent_version", "1 as version", "concept_name", "cast(1 as boolean) as archived", *other_cols)
    df2.show()
    
    a = icd10_concept_ids.union(df2).dropDuplicates()

    return a

    return left_join_cte

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
        #print(var_name,c1_percent,c2_percent,c1_count,c2_count,n1,n2)
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
        #print(f"{c} len={len(cohorts[c])}")

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
            #print(kind,v)
            var_name = v
            k = v   # TODO: Description of the variable
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
    df_table1 = pd.DataFrame(columns=['varname','characteristic',cohort_names[0],cohort_names[1],'SMD'])
    for i, row in table1.iterrows():
        varname = row.var_text
        if row_names == None:
            df_table1.loc[i] = [varname, varname, row.text_str1, row.text_str2, f"{row.sdiff:.3f}"]
        else:
            if varname in row_names:
                desc = row_names[varname]
            else:
                desc = varname
            df_table1.loc[i] = [row.var_text, desc, row.text_str1, row.text_str2, f"{row.sdiff:.3f}"]

    sdf = spark.createDataFrame(df_table1)
    return sdf

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.09dac4f7-282d-471c-ab3b-f0a50b0c1194"),
    Death_or_last_followup_date=Input(rid="ri.foundry.main.dataset.67ce4c32-187e-4f40-af6d-cb0a3d62ee2a"),
    compute_inclusion=Input(rid="ri.foundry.main.dataset.fe2ec78d-f56c-40f8-9aa1-6778020a8bdd"),
    fact_table_final=Input(rid="ri.foundry.main.dataset.78944dcb-af75-49bc-b8e3-1f041c3e6c25"),
    fact_table_pre=Input(rid="ri.foundry.main.dataset.3b11915c-f86b-43fc-bdfc-0cc54792963e"),
    make_cohort=Input(rid="ri.foundry.main.dataset.8bfe3e9a-d7d1-4d70-a22f-f295aa090587")
)
def attrition_table( compute_inclusion, fact_table_pre, make_cohort, fact_table_final, Death_or_last_followup_date):
    import pandas as pd
    #     sql = f"case when ((coalesce(office_visits_0_6,0) > 0) and (coalesce(office_visits_6_12,0) > 0)) and (coalesce(metformin_before_365,0) = 0) and (coalesce(comparator_meds_all_before_365,0) = 0) and ( (coalesce(metformin_after,0) = 1) or (coalesce(comparator_meds_all_after,0) = 1) ) then 1 else 0 end"

    df = make_cohort

    df = df.withColumn("excl_metformin_before_365",F.col("metformin_before_365")) 
    df = df.withColumn("excl_comparator_meds_all_before_365",F.col("comparator_meds_all_before_365"))


    strata = ['office_visits_0_6','office_visits_6_12','excl_metformin_before_365','excl_comparator_meds_all_before_365','excl_hosp_event','excl_ckd45'] #,'excl_death','excl_pasc'] # ,'metformin_after','comparator_meds_all_after'] 
    

    items = [[0,'cohort',df.count()]]
    i = 1
    for s in strata:
        if s.startswith('excl_'):
            f = f"(coalesce({s},0) = 0)"
        else:
            f = f"({s} > 0)"
        print(f"filter = {f}")
        c = df.filter(f)
        l = c.count()
        print(f"{s} = ",l)
        items += [[i,s,l]]
        i += 1
        df = c

    # How many in the OR clause
    o1 = df.filter(F.expr("(coalesce(metformin_after,0) = 1)")).count()
    print(f"metformin_after clause = ",o1)
    items += [[i,'metformin_after',o1]]
    i += 1
    o1 = df.filter(F.expr("(coalesce(comparator_meds_all_after,0) = 1)")).count()
    print(f"comparator_meds_all_after clause = ",o1)
    items += [[i,'comparator_meds_all_after',o1]]
    i += 1

    prior = l
    # How many included after cohorts determined
    l = compute_inclusion.filter(F.expr('included = 1')).count()
    print(f"included = ",l)
    items += [[i,'included',l]]
    i += 1

    for cohort in ['metformin','control']:
        c = fact_table_pre.filter(f"cohort = '{cohort}'").count()
        items += [[i,f"{cohort} cohort count",c]]
        i += 1

    # # How many after final inclusion criteria
    exclusions = ["site_no_u099","excl_pasc","excl_death"]
    filters = []
    for e in exclusions:
        filters.append(f"(coalesce({e},0) = 0)")
        filter_st = " and ".join(filters)
        print(filter_st)
        for cohort in ['metformin','control']: 
            c = fact_table_pre.filter(f"(cohort = '{cohort}') and {filter_st}").count()
            items += [[i,f"{cohort} cohort count exclude {e}",c]]
            i += 1

    print(f"items = {items}")

    df = pd.DataFrame(items,columns=['step','variable','records'])
    return df
    
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.fe2ec78d-f56c-40f8-9aa1-6778020a8bdd"),
    make_cohort=Input(rid="ri.foundry.main.dataset.8bfe3e9a-d7d1-4d70-a22f-f295aa090587")
)
def compute_inclusion(make_cohort):
    df = make_cohort

    sql = f"case when ((coalesce(office_visits_0_6,0) > 0) and (coalesce(office_visits_6_12,0) > 0)) and (coalesce(metformin_before_365,0) = 0) and (coalesce(comparator_meds_all_before_365,0) = 0) and ( (coalesce(metformin_after,0) = 1) or (coalesce(comparator_meds_all_after,0) = 1) ) and (coalesce(excl_hosp_event,0) = 0) and (coalesce(excl_ckd45,0) = 0) then 1 else 0 end"

    df = df.withColumn("included", F.expr(sql) )

    df = df.filter("included = 1")

    return df

    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.78944dcb-af75-49bc-b8e3-1f041c3e6c25"),
    fact_table_pre=Input(rid="ri.foundry.main.dataset.3b11915c-f86b-43fc-bdfc-0cc54792963e")
)
def fact_table_final(fact_table_pre):
    df = fact_table_pre

    exclusions = ["site_no_u099","excl_pasc",""]
    
    sql = "(coalesce(site_no_u099,0) = 0) and (coalesce(excl_pasc,0) = 0) and (coalesce(excl_death,0) = 0)"
    print(sql)

    df = df.filter(F.expr(sql))

    return df
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.3b11915c-f86b-43fc-bdfc-0cc54792963e"),
    COVID_Patient_Summary_Table=Input(rid="ri.foundry.main.dataset.e791e388-8c18-4d38-ae8a-9e8b3a91d556"),
    Death_or_last_followup_date=Input(rid="ri.foundry.main.dataset.67ce4c32-187e-4f40-af6d-cb0a3d62ee2a"),
    compute_inclusion=Input(rid="ri.foundry.main.dataset.fe2ec78d-f56c-40f8-9aa1-6778020a8bdd"),
    date_model=Input(rid="ri.foundry.main.dataset.f118a0c5-0a96-49f1-a497-021a3a656107"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.ec252b05-8f82-4f7f-a227-b3bb9bc578ef"),
    make_variables_a0=Input(rid="ri.foundry.main.dataset.0ec8b130-dd94-481e-b0a2-28ef355ffba4"),
    make_variables_a1=Input(rid="ri.foundry.main.dataset.68bedbe3-573b-454f-9212-65b7e96255a8"),
    make_variables_a2=Input(rid="ri.foundry.main.dataset.13a2b1f5-2af7-4175-86bb-ed70e8f233e1"),
    make_variables_a3=Input(rid="ri.foundry.main.dataset.57c710c1-b878-4537-b40b-d40d7190ba96"),
    person=Input(rid="ri.foundry.main.dataset.50cae11a-4afb-457d-99d4-55b4bc2cbe66"),
    visit_occurrence=Input(rid="ri.foundry.main.dataset.911d0bb2-c56e-46bd-af4f-8d9611183bb7")
)
def fact_table_pre(make_variables_a1, make_variables_a0, person, visit_occurrence, drug_exposure, COVID_Patient_Summary_Table, compute_inclusion, make_variables_a2, make_variables_a3, death, date_model, Death_or_last_followup_date):

    
    merge = compute_inclusion

    cps = [make_variables_a0, make_variables_a1, make_variables_a2, make_variables_a3 ]

    cols = ["age", "cohort_end_date", "index_date", "person_id_orig"]
    
    # Merge the variables together into a single fact table
    for cp in cps:
        merge = merge.join(cp.drop(*cols), "cohort_person_id", how="left")

    df = merge
    
    # Add demographics

    ll = COVID_Patient_Summary_Table.selectExpr("person_id","race_ethnicity","sex","Severity_Type","LC_u09_computable_phenotype_threshold_75","data_partner_id")
    df = df.join(ll,ll.person_id == df.person_id_orig)

    df = df.withColumn("Harmonized_gender", 
                                            F.when(F.col("sex") == "FEMALE", "Female")
                                            .when(F.col("sex") == "MALE", "Male")
                                            .when(F.col("sex").isNull(), "Missing")
                                            .otherwise(F.lit("Other/Unknown"))                                            
                                            )
    ## age
    df = df.withColumn("Harmonized_age_group", 
                                            F.when(F.col("age").isNull(), "09  No_age_information")
                                            .when(F.col("age") < 21, "01  0 to 21")
                                            .when(F.col("age") <= 45, "02  21 to 45")
                                            .when(F.col("age") <= 65, "03  46 to 65")
                                            .otherwise(F.lit("04  66 plus"))                                            
                                            )

    # BMI category
    df = df.withColumn("bmi_category",
            F.expr("""CASE WHEN bmi < 18.5 THEN '01  0 to 18.5'
                    WHEN bmi < 25 THEN '02  18.5 to 24.9'
                    WHEN bmi < 30 THEN '03  25.0 to 29.9'
                    WHEN bmi >= 30 THEN '04  30.0 plus'
                    ELSE '09  Other' end"""
                )
    )

    # COVID Era
    df = df.withColumn("covid_era",
            F.expr("""CASE WHEN index_date between to_date('2020-03-01') AND to_date('2020-09-30') THEN '01  Ancestral'
                    WHEN index_date between to_date('2020-10-01') AND to_date('2021-06-30') THEN '02  Alpha'
                    WHEN index_date between to_date('2021-07-01') AND to_date('2021-11-30') THEN '03  Delta'
                    WHEN index_date > to_date('2021-12-01') THEN '04  Omicron'
                    ELSE '09  Other' end"""
                )
    )

    # Compute control2 days from index
    vars = ["metformin", "ivermectin", "fluvoxamine", "montelukast", "albuterol", "fluticasone", "prednisone", "molnupiravir"]
    for v in vars:
        df = df.withColumn(f"{v}_days_from_index",
                F.expr(f"""datediff({v}_after_date, index_date)""")
        )
    
    control_list = ",".join([f"{x}_days_from_index" for x in ["ivermectin", "fluvoxamine", "montelukast", "fluticasone",]])
    df = df.withColumn("control_days_from_index",F.expr(f"least({control_list})"))

    for v in vars + ["control"]:
        df = df.withColumn(f"{v}_days_from_index_category",
            F.expr(f"""CASE WHEN {v}_days_from_index <= 0 THEN '01  neg 2 to 0'
                    WHEN {v}_days_from_index <= 3 THEN '02  1 to 3'
                    WHEN {v}_days_from_index <= 5 THEN '03  3 to 5'
                    WHEN {v}_days_from_index <= 7 THEN '04  5 to 7'
                    WHEN {v}_days_from_index <= 10 THEN '05  7 to 10'
                    WHEN {v}_days_from_index <= 14 THEN '06  10 to 14'
                    ELSE '09  14 plus' end"""
                )
            )

    # Label the cohorts
    df = df.withColumn("cohort",F.expr("case when (metformin_days_from_index is not null and metformin_days_from_index <= coalesce(control_days_from_index,999)) then 'metformin' when (control_days_from_index is not null and control_days_from_index < coalesce(metformin_days_from_index,999)) then 'control' else null end"))

    # combined_days_from_index
    df = df.withColumn("combined_days_from_index",F.expr("case when cohort = 'metformin' then metformin_days_from_index when cohort = 'control' then control_days_from_index else null end"))

    # egfr
    df = df.withColumn("egfr",F.expr(f"case when creatinine_before_mean <= 0 then null when sex = 'FEMALE' then 142 * pow(least(creatinine_before_mean/0.7,1),(-0.241)) * pow(greatest(creatinine_before_mean/0.7,1),(-1.2)) * pow(0.9938,age) * 1.012 when sex = 'MALE' then 142 * pow(least(creatinine_before_mean/0.9,1),(-0.302)) * pow(greatest(creatinine_before_mean/0.9,1),(-1.2)) * pow(0.9938,age) else null end"))

    # Data partner information
    # Sites that have never provided a U09.9 DX
    sites = '()' # site list censored
    df = df.withColumn("site_no_u099",F.expr(f"case when data_partner_id in {sites} then 1 else 0 end"))

    # Sites with high BMI missingness
    sites = '()' # site list censored
    df = df.withColumn("site_high_bmi_missing", F.expr(f"case when data_partner_id in {sites} then 1 else 0 end"))

    #
    # Censor and TTE variables
    #

    # treatment_index_date
    df = df.withColumn("treatment_index_date",F.expr("case when cohort = 'metformin' then metformin_after_date when cohort = 'control' then date_add(index_date,control_days_from_index) else null end"))

    # get PASC ML Model probability date   
    dm = date_model.selectExpr("person_id as person_id_orig","lc_date_offset","preponderance_date")
    df = df.join(dm,"person_id_orig",how="left" )
    df = df.withColumn("pasc_ml_date",F.expr("case when preponderance_date > treatment_index_date then preponderance_date else null end"))
    # PASC computable phenotype (ML)
    df = df.withColumn("pasc_ml",
                 F.expr("""case when pasc_ml_date is not null then 1 else 0 end"""))
    
    # max followup date
    p3 = Death_or_last_followup_date.selectExpr("person_id as person_id_orig","End_date_for_survival_analysis as death_last_fu_date", "death_info as deceased_yn")
    df = df.join(p3,"person_id_orig",how="left")

    df = df.withColumn("excl_death", F.expr("case when (deceased_yn = 1) and (death_last_fu_date <= treatment_index_date) then 1 else 0 end"))
    df = df.withColumn("excl_pasc", F.expr("case when (pasc_date is not null) and (pasc_date <= treatment_index_date) then 1 else 0 end"))
    df = df.withColumn("pasc_any", F.expr("case when pasc = 1 or pasc_ml = 1 then 1 else 0 end"))
    df = df.withColumn("pasc_any_date", F.expr("case when pasc_any = 1 then least(pasc_date,pasc_ml_date) else null end"))
    df = df.withColumn("pasc_any_days_from_index",F.expr("case when pasc_any = 1 then datediff(pasc_any_date,treatment_index_date) else null end"))
    
    df = df.withColumn("End_date_for_survival_analysis", F.expr("case when pasc_any = 1 then pasc_any_date else death_last_fu_date end"))
    df = df.withColumn("time_to_censor_days", F.expr("datediff(End_date_for_survival_analysis, treatment_index_date)"))
    df = df.withColumn("time_to_censor", F.expr("case when (time_to_censor_days is null) or (time_to_censor_days > 180) then 181 else time_to_censor_days end"))
    df = df.withColumn("outcome", F.expr("case when pasc_any = 1 and pasc_any_days_from_index > 180 then 0 else pasc_any end"))
    df = df.withColumn("censor_type", F.expr("case when outcome = 1 then 'pasc' when deceased_yn = 1 and pasc_any != 1 then 'death' else 'censored' end"))
    df = df.withColumn("death_180", F.expr("case when deceased_yn = 1 and time_to_censor_days <= 180 then 1 else 0 end"))
    df = df.withColumn("censor_type", F.expr("case when outcome = 1 then 'pasc' when death_180 = 1 and outcome != 1 then 'death' else 'censored' end"))
    df = df.withColumn("combined_outcome", F.expr("case when outcome = 1 or death_180 = 1 then 1 else 0 end"))
    df = df.withColumn("combined_time_to_censor",F.col("time_to_censor"))
    df = df.withColumn("combined_censor_type", F.expr("case when combined_outcome = 1 then 'outcome' else 'censored' end"))

    df.printSchema()

    # Fill in binary variables that are null with 0

    all_cols = inclusion_bin_cols + variables_bin_cols
    bin_dict = dict.fromkeys(all_cols, 0)
    df = df.fillna(bin_dict) 
    print("all columns = ",df.columns)

    # Exclude if not in a cohort 
    df = df.filter(F.expr("(cohort is not null)"))  

    return df
    

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.a54d8682-9007-4032-824c-e3d5eada0f64"),
    fact_table_final=Input(rid="ri.foundry.main.dataset.78944dcb-af75-49bc-b8e3-1f041c3e6c25")
)
def histogram_final(fact_table_final):
    import pandas as pd
    import matplotlib.pyplot as plt

    for cohort in ['metformin','control']:
        print(f"Cohort: {cohort}")
        df = fact_table_final.filter(f"(cohort = '{cohort}') and ({cohort}_days_from_index is not null and {cohort}_days_from_index <= 14)")
        
        df = df.toPandas()
        print("N = ",len(df))
        print("min = ",df[f"{cohort}_days_from_index"].min())
        print("max = ",df[f"{cohort}_days_from_index"].max())

        fig, ax = plt.subplots(figsize=(10,6))
        bins = list(range(-2,16))
        print("bins = ",bins)
        counts, rbins, _ = plt.hist(df[f'{cohort}_days_from_index'], bins=bins, edgecolor='black')
        print("counts = ",counts)
        print("rbins = ",rbins)
        for x,y in zip(bins,counts):
            plt.text(x+0.2, y+3, int(y), fontsize=10)
        plt.xticks(bins)
        ax.set_xticklabels([])
        for i, xi in enumerate(bins[:-1]):
            plt.text(xi+.4, -9, bins[i], fontsize=10)

        plt.title(f'{cohort}')
        plt.xlabel('Days from Index')
        plt.ylabel(f'Count')
        plt.grid(True)
        plt.show()


@transform_pandas(
    Output(rid="ri.vector.main.execute.ae7c3c82-e9da-4c23-a629-5283e9a7d4fb"),
    fact_table_pre=Input(rid="ri.foundry.main.dataset.3b11915c-f86b-43fc-bdfc-0cc54792963e")
)
def histogram_pre(fact_table_pre):
    import pandas as pd
    import matplotlib.pyplot as plt

    for cohort in ['metformin','control']:
        print(f"Cohort: {cohort}")
        df = fact_table_pre.filter(f"(cohort = '{cohort}') and ({cohort}_days_from_index is not null and {cohort}_days_from_index <= 14)")
        
        df = df.toPandas()
        print("N = ",len(df))
        print("min = ",df[f"{cohort}_days_from_index"].min())
        print("max = ",df[f"{cohort}_days_from_index"].max())

        fig, ax = plt.subplots(figsize=(10,6))
        bins = list(range(-2,16))
        print("bins = ",bins)
        counts, rbins, _ = plt.hist(df[f'{cohort}_days_from_index'], bins=bins, edgecolor='black')
        print("counts = ",counts)
        print("rbins = ",rbins)
        for x,y in zip(bins,counts):
            plt.text(x+0.2, y+3, int(y), fontsize=10)
        plt.xticks(bins)
        ax.set_xticklabels([])
        for i, xi in enumerate(bins[:-1]):
            plt.text(xi+.4, -9, bins[i], fontsize=10)

        plt.title(f'{cohort}')
        plt.xlabel('Days from Index')
        plt.ylabel(f'Count')
        plt.grid(True)
        plt.show()


@transform_pandas(
    Output(rid="ri.foundry.main.dataset.8bfe3e9a-d7d1-4d70-a22f-f295aa090587"),
    COVID_Patient_Summary_Table=Input(rid="ri.foundry.main.dataset.e791e388-8c18-4d38-ae8a-9e8b3a91d556"),
    concept=Input(rid="ri.foundry.main.dataset.5cb3c4a3-327a-47bf-a8bf-daf0cafe6772"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.900fa2ad-87ea-4285-be30-c6b5bab60e86"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.ec252b05-8f82-4f7f-a227-b3bb9bc578ef"),
    measurement=Input(rid="ri.foundry.main.dataset.d6054221-ee0c-4858-97de-22292458fa19"),
    my_codesets=Input(rid="ri.foundry.main.dataset.ac828b98-2947-43c3-98f9-f3f0fd9c3ff0"),
    observation=Input(rid="ri.foundry.main.dataset.b998b475-b229-471c-800e-9421491409f3"),
    person=Input(rid="ri.foundry.main.dataset.50cae11a-4afb-457d-99d4-55b4bc2cbe66"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.f6f0b5e0-a105-403a-a98f-0ee1c78137dc"),
    visit_occurrence=Input(rid="ri.foundry.main.dataset.911d0bb2-c56e-46bd-af4f-8d9611183bb7")
)
def make_cohort(condition_occurrence, drug_exposure, measurement, person, visit_occurrence, procedure_occurrence, observation, concept_set_members, death , concept, COVID_Patient_Summary_Table, my_codesets):

    # Use RECOVER COVID positive patients as the initial population (because that is who we have ML PASC computed)
    cohort = COVID_Patient_Summary_Table  
    cohort = cohort.selectExpr("person_id cohort_person_id", "person_id person_id_orig", "COVID_first_PCR_or_AG_lab_positive index_date", "age_at_covid age", "cast(null as date) as cohort_end_date")

    # Use our version of the codesets
    concept_set_members = my_codesets

    left_join_cte = cohort

    cte_1 = cohort.alias('cohort').join(visit_occurrence.alias('visit_occurrence'), cohort.person_id_orig == visit_occurrence.person_id, how='left').\
    where("((cohort_end_date is null) or (visit_start_date <= cohort_end_date)) AND visit_concept_id in (5083, 8883, 9202, 581476, 581477) AND (datediff(visit_start_date, index_date) >= -180) AND (datediff(visit_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "visit_start_date AS var_date",
    "visit_concept_id AS concept_id",
    "abs( dense_rank() over (partition by cohort_person_id order by visit_start_date) - dense_rank() over (partition by cohort_person_id order by visit_start_date desc) ) + 1 AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(visit_start_date, index_date) ASC) AS rn",
    )
    office_visits_0_6_cte = cte_1.selectExpr("cohort_person_id","var_val AS office_visits_0_6" ).where("rn == 1")
    left_join_cte = left_join_cte.join(office_visits_0_6_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(visit_occurrence.alias('visit_occurrence'), cohort.person_id_orig == visit_occurrence.person_id, how='left').\
    where("((cohort_end_date is null) or (visit_start_date <= cohort_end_date)) AND visit_concept_id in (5083, 8883, 9202, 581476, 581477) AND (datediff(visit_start_date, index_date) >= -365) AND (datediff(visit_start_date, index_date) <= -180)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "visit_start_date AS var_date",
    "visit_concept_id AS concept_id",
    "abs( dense_rank() over (partition by cohort_person_id order by visit_start_date) - dense_rank() over (partition by cohort_person_id order by visit_start_date desc) ) + 1 AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(visit_start_date, index_date) ASC) AS rn",
    )
    office_visits_6_12_cte = cte_1.selectExpr("cohort_person_id","var_val AS office_visits_6_12" ).where("rn == 1")
    left_join_cte = left_join_cte.join(office_visits_6_12_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(visit_occurrence.alias('visit_occurrence'), cohort.person_id_orig == visit_occurrence.person_id, how='left').\
    where("((cohort_end_date is null) or (visit_start_date <= cohort_end_date)) AND visit_concept_id in (5083, 8883, 9202, 581476, 581477) AND (datediff(visit_start_date, index_date) >= 0) AND (datediff(visit_start_date, index_date) <= 365)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "visit_start_date AS var_date",
    "visit_concept_id AS concept_id",
    "abs( dense_rank() over (partition by cohort_person_id order by visit_start_date) - dense_rank() over (partition by cohort_person_id order by visit_start_date desc) ) + 1 AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(visit_start_date, index_date) ASC) AS rn",
    )
    office_visits_after_cte = cte_1.selectExpr("cohort_person_id","var_val AS office_visits_after" ).where("rn == 1")
    left_join_cte = left_join_cte.join(office_visits_after_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "677036102"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -365) AND (datediff(drug_exposure_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    metformin_before_365_cte = cte_1.selectExpr("cohort_person_id","var_val AS metformin_before_365" ).where("rn == 1")
    left_join_cte = left_join_cte.join(metformin_before_365_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "677036102"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -180) AND (datediff(drug_exposure_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    metformin_before_180_cte = cte_1.selectExpr("cohort_person_id","var_val AS metformin_before_180" ).where("rn == 1")
    left_join_cte = left_join_cte.join(metformin_before_180_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "677036102"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -2) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    metformin_after_cte = cte_1.selectExpr("cohort_person_id","var_val AS metformin_after" ).where("rn == 1")
    left_join_cte = left_join_cte.join(metformin_after_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "677036102"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -2) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "drug_concept_id AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    metformin_after_date_cte = cte_1.selectExpr("cohort_person_id","var_date AS metformin_after_date" ).where("rn == 1")
    left_join_cte = left_join_cte.join(metformin_after_date_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "409971724"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -2) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    ivermectin_after_cte = cte_1.selectExpr("cohort_person_id","var_val AS ivermectin_after" ).where("rn == 1")
    left_join_cte = left_join_cte.join(ivermectin_after_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "762254277"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -2) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    fluvoxamine_after_cte = cte_1.selectExpr("cohort_person_id","var_val AS fluvoxamine_after" ).where("rn == 1")
    left_join_cte = left_join_cte.join(fluvoxamine_after_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "858141871"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -2) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    montelukast_after_cte = cte_1.selectExpr("cohort_person_id","var_val AS montelukast_after" ).where("rn == 1")
    left_join_cte = left_join_cte.join(montelukast_after_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "947341641"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -2) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    albuterol_after_cte = cte_1.selectExpr("cohort_person_id","var_val AS albuterol_after" ).where("rn == 1")
    left_join_cte = left_join_cte.join(albuterol_after_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "751977747"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -2) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    fluticasone_after_cte = cte_1.selectExpr("cohort_person_id","var_val AS fluticasone_after" ).where("rn == 1")
    left_join_cte = left_join_cte.join(fluticasone_after_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "892966630"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -2) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    prednisone_after_cte = cte_1.selectExpr("cohort_person_id","var_val AS prednisone_after" ).where("rn == 1")
    left_join_cte = left_join_cte.join(prednisone_after_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "156103933"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -2) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    molnupiravir_after_cte = cte_1.selectExpr("cohort_person_id","var_val AS molnupiravir_after" ).where("rn == 1")
    left_join_cte = left_join_cte.join(molnupiravir_after_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "409971724"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -2) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "drug_concept_id AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    ivermectin_after_date_cte = cte_1.selectExpr("cohort_person_id","var_date AS ivermectin_after_date" ).where("rn == 1")
    left_join_cte = left_join_cte.join(ivermectin_after_date_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "762254277"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -2) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "drug_concept_id AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    fluvoxamine_after_date_cte = cte_1.selectExpr("cohort_person_id","var_date AS fluvoxamine_after_date" ).where("rn == 1")
    left_join_cte = left_join_cte.join(fluvoxamine_after_date_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "858141871"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -2) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "drug_concept_id AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    montelukast_after_date_cte = cte_1.selectExpr("cohort_person_id","var_date AS montelukast_after_date" ).where("rn == 1")
    left_join_cte = left_join_cte.join(montelukast_after_date_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "947341641"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -2) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "drug_concept_id AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    albuterol_after_date_cte = cte_1.selectExpr("cohort_person_id","var_date AS albuterol_after_date" ).where("rn == 1")
    left_join_cte = left_join_cte.join(albuterol_after_date_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "751977747"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -2) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "drug_concept_id AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    fluticasone_after_date_cte = cte_1.selectExpr("cohort_person_id","var_date AS fluticasone_after_date" ).where("rn == 1")
    left_join_cte = left_join_cte.join(fluticasone_after_date_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "892966630"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -2) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "drug_concept_id AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    prednisone_after_date_cte = cte_1.selectExpr("cohort_person_id","var_date AS prednisone_after_date" ).where("rn == 1")
    left_join_cte = left_join_cte.join(prednisone_after_date_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "156103933"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -2) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "drug_concept_id AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    molnupiravir_after_date_cte = cte_1.selectExpr("cohort_person_id","var_date AS molnupiravir_after_date" ).where("rn == 1")
    left_join_cte = left_join_cte.join(molnupiravir_after_date_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(visit_occurrence.alias('visit_occurrence'), cohort.person_id_orig == visit_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == visit_occurrence.visit_concept_id) & (concept_set_members.codeset_id == "972465851"), how="inner").\
    where("((cohort_end_date is null) or (visit_start_date <= cohort_end_date)) AND (datediff(visit_start_date, index_date) >= -3) AND (datediff(visit_start_date, index_date) <= 1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "visit_start_date AS var_date",
    "visit_concept_id AS concept_id",
    "CASE WHEN visit_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(visit_start_date, index_date) ASC) AS rn",
    )
    excl_hosp_event_cte = cte_1.selectExpr("cohort_person_id","var_val AS excl_hosp_event" ).where("rn == 1")
    left_join_cte = left_join_cte.join(excl_hosp_event_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "comparator_meds_all"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -365) AND (datediff(drug_exposure_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    comparator_meds_all_before_365_cte = cte_1.selectExpr("cohort_person_id","var_val AS comparator_meds_all_before_365" ).where("rn == 1")
    left_join_cte = left_join_cte.join(comparator_meds_all_before_365_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "comparator_meds_all"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -2) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    comparator_meds_all_after_cte = cte_1.selectExpr("cohort_person_id","var_val AS comparator_meds_all_after" ).where("rn == 1")
    left_join_cte = left_join_cte.join(comparator_meds_all_after_cte, 'cohort_person_id', how='left')

    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "ckd45"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    excl_ckd45_cte = cte_1.selectExpr("cohort_person_id","var_val AS excl_ckd45" ).where("rn == 1")
    left_join_cte = left_join_cte.join(excl_ckd45_cte, 'cohort_person_id', how='left')

    return left_join_cte

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0ec8b130-dd94-481e-b0a2-28ef355ffba4"),
    COVID_Patient_Summary_Table=Input(rid="ri.foundry.main.dataset.e791e388-8c18-4d38-ae8a-9e8b3a91d556"),
    compute_inclusion=Input(rid="ri.foundry.main.dataset.fe2ec78d-f56c-40f8-9aa1-6778020a8bdd"),
    concept=Input(rid="ri.foundry.main.dataset.5cb3c4a3-327a-47bf-a8bf-daf0cafe6772"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.900fa2ad-87ea-4285-be30-c6b5bab60e86"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.ec252b05-8f82-4f7f-a227-b3bb9bc578ef"),
    measurement=Input(rid="ri.foundry.main.dataset.d6054221-ee0c-4858-97de-22292458fa19"),
    my_codesets=Input(rid="ri.foundry.main.dataset.ac828b98-2947-43c3-98f9-f3f0fd9c3ff0"),
    observation=Input(rid="ri.foundry.main.dataset.b998b475-b229-471c-800e-9421491409f3"),
    person=Input(rid="ri.foundry.main.dataset.50cae11a-4afb-457d-99d4-55b4bc2cbe66"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.f6f0b5e0-a105-403a-a98f-0ee1c78137dc"),
    visit_occurrence=Input(rid="ri.foundry.main.dataset.911d0bb2-c56e-46bd-af4f-8d9611183bb7")
)
def make_variables_a0(condition_occurrence, drug_exposure, measurement, person, visit_occurrence, procedure_occurrence, observation, concept_set_members, death , concept, COVID_Patient_Summary_Table, compute_inclusion, my_codesets):

    cohort = compute_inclusion
    cohort = cohort.selectExpr("cohort_person_id", "person_id_orig", "index_date index_date", "age", "cohort_end_date")

    # We need to break the variables into slices because Palantir fails if too many are computed at the same time

    # Use our version of the codesets
    concept_set_members = my_codesets
    left_join_cte = cohort

    #
    # Slice 0
    #
    # 0:weight
    cte_1 = cohort.alias('cohort').join(measurement.alias('measurement'), cohort.person_id_orig == measurement.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == measurement.measurement_concept_id) & (concept_set_members.codeset_id == "776390058"), how="inner").\
    where("((cohort_end_date is null) or (measurement_date <= cohort_end_date)) AND (datediff(measurement_date, index_date) >= -365) AND (datediff(measurement_date, index_date) <= 0)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "measurement_date AS var_date",
    "measurement_concept_id AS concept_id",
    "harmonized_value_as_number AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(measurement_date, index_date) DESC) AS rn",
    )
    weight_cte = cte_1.selectExpr("cohort_person_id","var_val AS weight" ).where("rn == 1")
    left_join_cte = left_join_cte.join(weight_cte, 'cohort_person_id', how='left')

    # 1:height
    cte_1 = cohort.alias('cohort').join(measurement.alias('measurement'), cohort.person_id_orig == measurement.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == measurement.measurement_concept_id) & (concept_set_members.codeset_id == "754731201"), how="inner").\
    where("((cohort_end_date is null) or (measurement_date <= cohort_end_date)) AND (datediff(measurement_date, index_date) >= -365) AND (datediff(measurement_date, index_date) <= 0)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "measurement_date AS var_date",
    "measurement_concept_id AS concept_id",
    "harmonized_value_as_number AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(measurement_date, index_date) DESC) AS rn",
    )
    height_cte = cte_1.selectExpr("cohort_person_id","var_val AS height" ).where("rn == 1")
    left_join_cte = left_join_cte.join(height_cte, 'cohort_person_id', how='left')

    # 2:bmi_computed
    cte_1 = cohort.alias("cohort").join(height_cte.alias("height"), height_cte.cohort_person_id == cohort.cohort_person_id, how="left").join(weight_cte.alias("weight"), weight_cte.cohort_person_id == cohort.cohort_person_id, how="left").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "null AS var_date",
    "null AS concept_id",
    "case when (height.height is not null and (height.height >= 0.6 and height.height <= 2.43)) and (weight.weight is not null and (weight.weight >= 5 and weight.weight <= 300)) then weight.weight / (height.height * height.height) else null end AS var_val",
    )
    bmi_computed_cte = cte_1.selectExpr("cohort_person_id","var_val AS bmi_computed" )
    left_join_cte = left_join_cte.join(bmi_computed_cte, 'cohort_person_id', how='left')

    # 3:bmi_site
    cte_1 = cohort.alias('cohort').join(measurement.alias('measurement'), cohort.person_id_orig == measurement.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == measurement.measurement_concept_id) & (concept_set_members.codeset_id == "65622096"), how="inner").\
    where("((cohort_end_date is null) or (measurement_date <= cohort_end_date)) AND (datediff(measurement_date, index_date) >= -365) AND (datediff(measurement_date, index_date) <= 0)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "measurement_date AS var_date",
    "measurement_concept_id AS concept_id",
    "harmonized_value_as_number AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(measurement_date, index_date) DESC) AS rn",
    )
    bmi_site_cte = cte_1.selectExpr("cohort_person_id","var_val AS bmi_site" ).where("rn == 1")
    left_join_cte = left_join_cte.join(bmi_site_cte, 'cohort_person_id', how='left')

    # 4:bmi
    cte_1 = cohort.alias("cohort").join(bmi_site_cte.alias("bmi_site"), bmi_site_cte.cohort_person_id == cohort.cohort_person_id, how="left").join(bmi_computed_cte.alias("bmi_computed"), bmi_computed_cte.cohort_person_id == cohort.cohort_person_id, how="left").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "null AS var_date",
    "null AS concept_id",
    "coalesce(bmi_site.bmi_site,bmi_computed.bmi_computed) AS var_val",
    )
    bmi_cte = cte_1.selectExpr("cohort_person_id","var_val AS bmi" )
    left_join_cte = left_join_cte.join(bmi_cte, 'cohort_person_id', how='left')

    # 5:deceased
    cte_1 = cohort.alias('cohort').join(death.alias('death'), cohort.person_id_orig == death.person_id, how='left').\
    where("((cohort_end_date is null) or (death_date <= cohort_end_date))").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "death_date AS var_date",
    "death_type_concept_id AS concept_id",
    "CASE WHEN death_type_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(death_date, index_date) DESC) AS rn",
    )
    deceased_cte = cte_1.selectExpr("cohort_person_id","var_val AS deceased" ).where("rn == 1")
    left_join_cte = left_join_cte.join(deceased_cte, 'cohort_person_id', how='left')

    # 6:ace_before
    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "783781794"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -365) AND (datediff(drug_exposure_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) DESC) AS rn",
    )
    ace_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS ace_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(ace_before_cte, 'cohort_person_id', how='left')

    # 7:arb_before
    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "364083331"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -365) AND (datediff(drug_exposure_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) DESC) AS rn",
    )
    arb_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS arb_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(arb_before_cte, 'cohort_person_id', how='left')

    # 8:statin_before
    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "143034569"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -365) AND (datediff(drug_exposure_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) DESC) AS rn",
    )
    statin_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS statin_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(statin_before_cte, 'cohort_person_id', how='left')

    # 9:anticoagulant_before
    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "761556952"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -365) AND (datediff(drug_exposure_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) DESC) AS rn",
    )
    anticoagulant_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS anticoagulant_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(anticoagulant_before_cte, 'cohort_person_id', how='left')

    # 10:aspirin_before
    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "334118105"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -365) AND (datediff(drug_exposure_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) DESC) AS rn",
    )
    aspirin_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS aspirin_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(aspirin_before_cte, 'cohort_person_id', how='left')

    # 11:cad
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "630858234"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    cad_cte = cte_1.selectExpr("cohort_person_id","var_val AS cad" ).where("rn == 1")
    left_join_cte = left_join_cte.join(cad_cte, 'cohort_person_id', how='left')

    # 12:cancer
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "535274723"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    cancer_cte = cte_1.selectExpr("cohort_person_id","var_val AS cancer" ).where("rn == 1")
    left_join_cte = left_join_cte.join(cancer_cte, 'cohort_person_id', how='left')

    # 13:ckd123
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "451190647"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    ckd123_cte = cte_1.selectExpr("cohort_person_id","var_val AS ckd123" ).where("rn == 1")
    left_join_cte = left_join_cte.join(ckd123_cte, 'cohort_person_id', how='left')

    # 14:heart_failure
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "875317226"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    heart_failure_cte = cte_1.selectExpr("cohort_person_id","var_val AS heart_failure" ).where("rn == 1")
    left_join_cte = left_join_cte.join(heart_failure_cte, 'cohort_person_id', how='left')

    # 15:hypertension
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "106701324"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    hypertension_cte = cte_1.selectExpr("cohort_person_id","var_val AS hypertension" ).where("rn == 1")
    left_join_cte = left_join_cte.join(hypertension_cte, 'cohort_person_id', how='left')

    # 16:liver_disease
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "882514953"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    liver_disease_cte = cte_1.selectExpr("cohort_person_id","var_val AS liver_disease" ).where("rn == 1")
    left_join_cte = left_join_cte.join(liver_disease_cte, 'cohort_person_id', how='left')

    # 17:t1dm_after
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "1000090029"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -2) AND (datediff(condition_start_date, index_date) <= 365)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    t1dm_after_cte = cte_1.selectExpr("cohort_person_id","var_val AS t1dm_after" ).where("rn == 1")
    left_join_cte = left_join_cte.join(t1dm_after_cte, 'cohort_person_id', how='left')

    # 18:t2dm_after
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "1000076523"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -2) AND (datediff(condition_start_date, index_date) <= 365)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    t2dm_after_cte = cte_1.selectExpr("cohort_person_id","var_val AS t2dm_after" ).where("rn == 1")
    left_join_cte = left_join_cte.join(t2dm_after_cte, 'cohort_person_id', how='left')

    # 19:prediabetes_after
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "690869593"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -2) AND (datediff(condition_start_date, index_date) <= 365)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    prediabetes_after_cte = cte_1.selectExpr("cohort_person_id","var_val AS prediabetes_after" ).where("rn == 1")
    left_join_cte = left_join_cte.join(prediabetes_after_cte, 'cohort_person_id', how='left')

    # 20:dm_complicated_after
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "18918743"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -2) AND (datediff(condition_start_date, index_date) <= 365)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    dm_complicated_after_cte = cte_1.selectExpr("cohort_person_id","var_val AS dm_complicated_after" ).where("rn == 1")
    left_join_cte = left_join_cte.join(dm_complicated_after_cte, 'cohort_person_id', how='left')

    # 21:dm_uncomplicated_after
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "248468138"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -2) AND (datediff(condition_start_date, index_date) <= 365)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    dm_uncomplicated_after_cte = cte_1.selectExpr("cohort_person_id","var_val AS dm_uncomplicated_after" ).where("rn == 1")
    left_join_cte = left_join_cte.join(dm_uncomplicated_after_cte, 'cohort_person_id', how='left')

    # 22:t1dm_after_date
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "1000090029"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -2) AND (datediff(condition_start_date, index_date) <= 365)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "condition_concept_id AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    t1dm_after_date_cte = cte_1.selectExpr("cohort_person_id","var_date AS t1dm_after_date" ).where("rn == 1")
    left_join_cte = left_join_cte.join(t1dm_after_date_cte, 'cohort_person_id', how='left')

    # 23:t2dm_after_date
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "1000076523"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -2) AND (datediff(condition_start_date, index_date) <= 365)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "condition_concept_id AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    t2dm_after_date_cte = cte_1.selectExpr("cohort_person_id","var_date AS t2dm_after_date" ).where("rn == 1")
    left_join_cte = left_join_cte.join(t2dm_after_date_cte, 'cohort_person_id', how='left')

    # 24:prediabetes_after_date
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "690869593"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -2) AND (datediff(condition_start_date, index_date) <= 365)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "condition_concept_id AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    prediabetes_after_date_cte = cte_1.selectExpr("cohort_person_id","var_date AS prediabetes_after_date" ).where("rn == 1")
    left_join_cte = left_join_cte.join(prediabetes_after_date_cte, 'cohort_person_id', how='left')

    return left_join_cte

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.68bedbe3-573b-454f-9212-65b7e96255a8"),
    COVID_Patient_Summary_Table=Input(rid="ri.foundry.main.dataset.e791e388-8c18-4d38-ae8a-9e8b3a91d556"),
    compute_inclusion=Input(rid="ri.foundry.main.dataset.fe2ec78d-f56c-40f8-9aa1-6778020a8bdd"),
    concept=Input(rid="ri.foundry.main.dataset.5cb3c4a3-327a-47bf-a8bf-daf0cafe6772"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.900fa2ad-87ea-4285-be30-c6b5bab60e86"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.ec252b05-8f82-4f7f-a227-b3bb9bc578ef"),
    measurement=Input(rid="ri.foundry.main.dataset.d6054221-ee0c-4858-97de-22292458fa19"),
    my_codesets=Input(rid="ri.foundry.main.dataset.ac828b98-2947-43c3-98f9-f3f0fd9c3ff0"),
    observation=Input(rid="ri.foundry.main.dataset.b998b475-b229-471c-800e-9421491409f3"),
    person=Input(rid="ri.foundry.main.dataset.50cae11a-4afb-457d-99d4-55b4bc2cbe66"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.f6f0b5e0-a105-403a-a98f-0ee1c78137dc"),
    visit_occurrence=Input(rid="ri.foundry.main.dataset.911d0bb2-c56e-46bd-af4f-8d9611183bb7")
)
def make_variables_a1(condition_occurrence, drug_exposure, measurement, person, visit_occurrence, procedure_occurrence, observation, concept_set_members, death , concept, COVID_Patient_Summary_Table, compute_inclusion, my_codesets):

    cohort = compute_inclusion
    cohort = cohort.selectExpr("cohort_person_id", "person_id_orig", "index_date index_date", "age", "cohort_end_date")

    # Use our version of the codesets
    concept_set_members = my_codesets
    left_join_cte = cohort

    #
    # Slice 1
    #
    # 25:dm_complicated_after_date
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "18918743"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -2) AND (datediff(condition_start_date, index_date) <= 365)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    dm_complicated_after_date_cte = cte_1.selectExpr("cohort_person_id","var_date AS dm_complicated_after_date" ).where("rn == 1")
    left_join_cte = left_join_cte.join(dm_complicated_after_date_cte, 'cohort_person_id', how='left')

    # 26:dm_uncomplicated_after_date
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "248468138"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -2) AND (datediff(condition_start_date, index_date) <= 365)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    dm_uncomplicated_after_date_cte = cte_1.selectExpr("cohort_person_id","var_date AS dm_uncomplicated_after_date" ).where("rn == 1")
    left_join_cte = left_join_cte.join(dm_uncomplicated_after_date_cte, 'cohort_person_id', how='left')

    # 27:pasc
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "708775231"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= 0) AND (datediff(condition_start_date, index_date) <= 180)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    pasc_cte = cte_1.selectExpr("cohort_person_id","var_val AS pasc" ).where("rn == 1")
    left_join_cte = left_join_cte.join(pasc_cte, 'cohort_person_id', how='left')

    # 28:hosp_outcome
    cte_1 = cohort.alias('cohort').join(visit_occurrence.alias('visit_occurrence'), cohort.person_id_orig == visit_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == visit_occurrence.visit_concept_id) & (concept_set_members.codeset_id == "972465851"), how="inner").\
    where("((cohort_end_date is null) or (visit_start_date <= cohort_end_date)) AND (datediff(visit_start_date, index_date) >= 0) AND (datediff(visit_start_date, index_date) <= 7)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "visit_start_date AS var_date",
    "visit_concept_id AS concept_id",
    "CASE WHEN visit_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(visit_start_date, index_date) ASC) AS rn",
    )
    hosp_outcome_cte = cte_1.selectExpr("cohort_person_id","var_val AS hosp_outcome" ).where("rn == 1")
    left_join_cte = left_join_cte.join(hosp_outcome_cte, 'cohort_person_id', how='left')

    # 29:nirmatrelvir
    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "399252964"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= 0) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    nirmatrelvir_cte = cte_1.selectExpr("cohort_person_id","var_val AS nirmatrelvir" ).where("rn == 1")
    left_join_cte = left_join_cte.join(nirmatrelvir_cte, 'cohort_person_id', how='left')

    # 30:ritonavir
    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "329050933"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= 0) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    ritonavir_cte = cte_1.selectExpr("cohort_person_id","var_val AS ritonavir" ).where("rn == 1")
    left_join_cte = left_join_cte.join(ritonavir_cte, 'cohort_person_id', how='left')

    # 31:paxlovid
    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "798981734"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= 0) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    paxlovid_cte = cte_1.selectExpr("cohort_person_id","var_val AS paxlovid" ).where("rn == 1")
    left_join_cte = left_join_cte.join(paxlovid_cte, 'cohort_person_id', how='left')

    # 32:remdesivir
    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "719693192"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= 0) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    remdesivir_cte = cte_1.selectExpr("cohort_person_id","var_val AS remdesivir" ).where("rn == 1")
    left_join_cte = left_join_cte.join(remdesivir_cte, 'cohort_person_id', how='left')

    # 33:dexamethasone
    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "485640400"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= 0) AND (datediff(drug_exposure_start_date, index_date) <= 14)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    dexamethasone_cte = cte_1.selectExpr("cohort_person_id","var_val AS dexamethasone" ).where("rn == 1")
    left_join_cte = left_join_cte.join(dexamethasone_cte, 'cohort_person_id', how='left')

    # 34:a1c_gt65_before
    cte_1 = cohort.alias('cohort').join(measurement.alias('measurement'), cohort.person_id_orig == measurement.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == measurement.measurement_concept_id) & (concept_set_members.codeset_id == "381434987"), how="inner").\
    where("((cohort_end_date is null) or (measurement_date <= cohort_end_date)) AND (datediff(measurement_date, index_date) >= -3650) AND (datediff(measurement_date, index_date) <= -1) AND (harmonized_value_as_number >= 6.5)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "measurement_date AS var_date",
    "measurement_concept_id AS concept_id",
    "CASE WHEN harmonized_value_as_number is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(measurement_date, index_date) ASC) AS rn",
    )
    a1c_gt65_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS a1c_gt65_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(a1c_gt65_before_cte, 'cohort_person_id', how='left')

    # 35:a1c_last
    cte_1 = cohort.alias('cohort').join(measurement.alias('measurement'), cohort.person_id_orig == measurement.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == measurement.measurement_concept_id) & (concept_set_members.codeset_id == "381434987"), how="inner").\
    where("((cohort_end_date is null) or (measurement_date <= cohort_end_date)) AND (datediff(measurement_date, index_date) >= -3650) AND (datediff(measurement_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "measurement_date AS var_date",
    "measurement_concept_id AS concept_id",
    "harmonized_value_as_number AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(measurement_date, index_date) DESC) AS rn",
    )
    a1c_last_cte = cte_1.selectExpr("cohort_person_id","var_val AS a1c_last" ).where("rn == 1")
    left_join_cte = left_join_cte.join(a1c_last_cte, 'cohort_person_id', how='left')

    # 36:nafld_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "305940281"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    nafld_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS nafld_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(nafld_before_cte, 'cohort_person_id', how='left')

    # 37:adjustment_disorder_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "adjustment_disorder"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    adjustment_disorder_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS adjustment_disorder_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(adjustment_disorder_before_cte, 'cohort_person_id', how='left')

    # 38:dysthymia_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "dysthymia"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    dysthymia_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS dysthymia_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(dysthymia_before_cte, 'cohort_person_id', how='left')

    # 39:seasonal_affective_disorder_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "seasonal_affective_disorder"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    seasonal_affective_disorder_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS seasonal_affective_disorder_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(seasonal_affective_disorder_before_cte, 'cohort_person_id', how='left')

    # 40:hypomania_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "hypomania"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    hypomania_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS hypomania_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(hypomania_before_cte, 'cohort_person_id', how='left')

    # 41:dysphoria_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "dysphoria"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    dysphoria_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS dysphoria_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(dysphoria_before_cte, 'cohort_person_id', how='left')

    # 42:insomnia_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "insomnia"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    insomnia_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS insomnia_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(insomnia_before_cte, 'cohort_person_id', how='left')

    # 43:bipolar_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "bipolar"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    bipolar_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS bipolar_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(bipolar_before_cte, 'cohort_person_id', how='left')

    # 44:mood_disorder_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "mood_disorder"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    mood_disorder_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS mood_disorder_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(mood_disorder_before_cte, 'cohort_person_id', how='left')

    # 45:steroids_before
    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "steroids"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -365) AND (datediff(drug_exposure_start_date, index_date) <= 0)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) DESC) AS rn",
    )
    steroids_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS steroids_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(steroids_before_cte, 'cohort_person_id', how='left')

    # 46:creatinine_before_mean
    cte_1 = cohort.alias('cohort').join(measurement.alias('measurement'), cohort.person_id_orig == measurement.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == measurement.measurement_concept_id) & (concept_set_members.codeset_id == "615348047"), how="inner").\
    where("((cohort_end_date is null) or (measurement_date <= cohort_end_date)) AND (datediff(measurement_date, index_date) >= -365) AND (datediff(measurement_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "measurement_date AS var_date",
    "measurement_concept_id AS concept_id",
    "avg(harmonized_value_as_number) OVER (PARTITION BY cohort_person_id) AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(measurement_date, index_date) ASC) AS rn",
    )
    creatinine_before_mean_cte = cte_1.selectExpr("cohort_person_id","var_val AS creatinine_before_mean" ).where("rn == 1")
    left_join_cte = left_join_cte.join(creatinine_before_mean_cte, 'cohort_person_id', how='left')

    # 47:creatinine_before_date
    cte_1 = cohort.alias('cohort').join(measurement.alias('measurement'), cohort.person_id_orig == measurement.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == measurement.measurement_concept_id) & (concept_set_members.codeset_id == "615348047"), how="inner").\
    where("((cohort_end_date is null) or (measurement_date <= cohort_end_date)) AND (datediff(measurement_date, index_date) >= -365) AND (datediff(measurement_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "measurement_date AS var_date",
    "measurement_concept_id AS concept_id",
    "harmonized_value_as_number AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(measurement_date, index_date) DESC) AS rn",
    )
    creatinine_before_date_cte = cte_1.selectExpr("cohort_person_id","var_date AS creatinine_before_date" ).where("rn == 1")
    left_join_cte = left_join_cte.join(creatinine_before_date_cte, 'cohort_person_id', how='left')

    # 48:binge_eating_disorder_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "binge_eating_disorder"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    binge_eating_disorder_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS binge_eating_disorder_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(binge_eating_disorder_before_cte, 'cohort_person_id', how='left')

    # 49:bulimia_nervosa_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "bulimia_nervosa"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    bulimia_nervosa_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS bulimia_nervosa_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(bulimia_nervosa_before_cte, 'cohort_person_id', how='left')

    
    return left_join_cte

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.13a2b1f5-2af7-4175-86bb-ed70e8f233e1"),
    COVID_Patient_Summary_Table=Input(rid="ri.foundry.main.dataset.e791e388-8c18-4d38-ae8a-9e8b3a91d556"),
    compute_inclusion=Input(rid="ri.foundry.main.dataset.fe2ec78d-f56c-40f8-9aa1-6778020a8bdd"),
    concept=Input(rid="ri.foundry.main.dataset.5cb3c4a3-327a-47bf-a8bf-daf0cafe6772"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.900fa2ad-87ea-4285-be30-c6b5bab60e86"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.ec252b05-8f82-4f7f-a227-b3bb9bc578ef"),
    measurement=Input(rid="ri.foundry.main.dataset.d6054221-ee0c-4858-97de-22292458fa19"),
    my_codesets=Input(rid="ri.foundry.main.dataset.ac828b98-2947-43c3-98f9-f3f0fd9c3ff0"),
    observation=Input(rid="ri.foundry.main.dataset.b998b475-b229-471c-800e-9421491409f3"),
    person=Input(rid="ri.foundry.main.dataset.50cae11a-4afb-457d-99d4-55b4bc2cbe66"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.f6f0b5e0-a105-403a-a98f-0ee1c78137dc"),
    visit_occurrence=Input(rid="ri.foundry.main.dataset.911d0bb2-c56e-46bd-af4f-8d9611183bb7")
)
def make_variables_a2(condition_occurrence, drug_exposure, measurement, person, visit_occurrence, procedure_occurrence, observation, concept_set_members, death , concept, COVID_Patient_Summary_Table, compute_inclusion, my_codesets):

    cohort = compute_inclusion
    cohort = cohort.selectExpr("cohort_person_id", "person_id_orig", "index_date index_date", "age", "cohort_end_date")

    # Use our version of the codesets
    concept_set_members = my_codesets
    left_join_cte = cohort

    #
    # Slice 2
    #
    # 50:generalized_anxiety_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "generalized_anxiety"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    generalized_anxiety_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS generalized_anxiety_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(generalized_anxiety_before_cte, 'cohort_person_id', how='left')

    # 51:major_depression_disorder_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "major_depression_disorder"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    major_depression_disorder_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS major_depression_disorder_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(major_depression_disorder_before_cte, 'cohort_person_id', how='left')

    # 52:ocd_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "126474015"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    ocd_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS ocd_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(ocd_before_cte, 'cohort_person_id', how='left')

    # 53:panic_disorder_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "panic_disorder"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    panic_disorder_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS panic_disorder_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(panic_disorder_before_cte, 'cohort_person_id', how='left')

    # 54:ptsd_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "ptsd"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    ptsd_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS ptsd_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(ptsd_before_cte, 'cohort_person_id', how='left')

    # 55:social_anxiety_disorder_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "social_anxiety_disorder"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    social_anxiety_disorder_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS social_anxiety_disorder_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(social_anxiety_disorder_before_cte, 'cohort_person_id', how='left')

    # 56:asthma_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "692562246"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    asthma_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS asthma_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(asthma_before_cte, 'cohort_person_id', how='left')

    # 57:copd_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "903690033"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    copd_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS copd_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(copd_before_cte, 'cohort_person_id', how='left')

    # 58:bronchiolitis_obliterans_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "bronchiolitis_obliterans"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    bronchiolitis_obliterans_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS bronchiolitis_obliterans_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(bronchiolitis_obliterans_before_cte, 'cohort_person_id', how='left')

    # 59:eosinophilic_esophagitis_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "eosinophilic_esophagitis"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    eosinophilic_esophagitis_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS eosinophilic_esophagitis_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(eosinophilic_esophagitis_before_cte, 'cohort_person_id', how='left')

    # 60:phct_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "phct"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    phct_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS phct_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(phct_before_cte, 'cohort_person_id', how='left')

    # 61:mast_cell_activation_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "mast_cell_activation"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    mast_cell_activation_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS mast_cell_activation_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(mast_cell_activation_before_cte, 'cohort_person_id', how='left')

    # 62:exercise_induced_asthma_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "exercise_induced_asthma"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    exercise_induced_asthma_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS exercise_induced_asthma_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(exercise_induced_asthma_before_cte, 'cohort_person_id', how='left')

    # 63:exercise_induced_bronchoconstriction_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "exercise_induced_bronchoconstriction"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    exercise_induced_bronchoconstriction_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS exercise_induced_bronchoconstriction_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(exercise_induced_bronchoconstriction_before_cte, 'cohort_person_id', how='left')

    # 64:allergic_rhinitis_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "allergic_rhinitis"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    allergic_rhinitis_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS allergic_rhinitis_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(allergic_rhinitis_before_cte, 'cohort_person_id', how='left')

    # 65:prediabetes_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "690869593"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    prediabetes_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS prediabetes_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(prediabetes_before_cte, 'cohort_person_id', how='left')

    # 66:t1dm_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "1000090029"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    t1dm_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS t1dm_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(t1dm_before_cte, 'cohort_person_id', how='left')

    # 67:t2dm_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "1000076523"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    t2dm_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS t2dm_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(t2dm_before_cte, 'cohort_person_id', how='left')

    # 68:gestational_dm_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "382227453"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    gestational_dm_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS gestational_dm_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(gestational_dm_before_cte, 'cohort_person_id', how='left')

    # 69:antipsychotic_induced_weight_gain_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "antipsychotic_induced_weight_gain"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    antipsychotic_induced_weight_gain_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS antipsychotic_induced_weight_gain_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(antipsychotic_induced_weight_gain_before_cte, 'cohort_person_id', how='left')

    # 70:pcos_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "349696666"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    pcos_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS pcos_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(pcos_before_cte, 'cohort_person_id', how='left')

    # 71:ckd4_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "778290097"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    ckd4_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS ckd4_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(ckd4_before_cte, 'cohort_person_id', how='left')

    # 72:esrd_dialysis_before
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "336612223"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= -3650) AND (datediff(condition_start_date, index_date) <= -3)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    esrd_dialysis_before_cte = cte_1.selectExpr("cohort_person_id","var_val AS esrd_dialysis_before" ).where("rn == 1")
    left_join_cte = left_join_cte.join(esrd_dialysis_before_cte, 'cohort_person_id', how='left')

    # 73:biguanides_before_365
    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "267122637"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -365) AND (datediff(drug_exposure_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    biguanides_before_365_cte = cte_1.selectExpr("cohort_person_id","var_val AS biguanides_before_365" ).where("rn == 1")
    left_join_cte = left_join_cte.join(biguanides_before_365_cte, 'cohort_person_id', how='left')

    # 74:dpp4_before_365
    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "82538978"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -365) AND (datediff(drug_exposure_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    dpp4_before_365_cte = cte_1.selectExpr("cohort_person_id","var_val AS dpp4_before_365" ).where("rn == 1")
    left_join_cte = left_join_cte.join(dpp4_before_365_cte, 'cohort_person_id', how='left')

    return left_join_cte

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.57c710c1-b878-4537-b40b-d40d7190ba96"),
    COVID_Patient_Summary_Table=Input(rid="ri.foundry.main.dataset.e791e388-8c18-4d38-ae8a-9e8b3a91d556"),
    compute_inclusion=Input(rid="ri.foundry.main.dataset.fe2ec78d-f56c-40f8-9aa1-6778020a8bdd"),
    concept=Input(rid="ri.foundry.main.dataset.5cb3c4a3-327a-47bf-a8bf-daf0cafe6772"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.900fa2ad-87ea-4285-be30-c6b5bab60e86"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.ec252b05-8f82-4f7f-a227-b3bb9bc578ef"),
    measurement=Input(rid="ri.foundry.main.dataset.d6054221-ee0c-4858-97de-22292458fa19"),
    my_codesets=Input(rid="ri.foundry.main.dataset.ac828b98-2947-43c3-98f9-f3f0fd9c3ff0"),
    observation=Input(rid="ri.foundry.main.dataset.b998b475-b229-471c-800e-9421491409f3"),
    person=Input(rid="ri.foundry.main.dataset.50cae11a-4afb-457d-99d4-55b4bc2cbe66"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.f6f0b5e0-a105-403a-a98f-0ee1c78137dc"),
    visit_occurrence=Input(rid="ri.foundry.main.dataset.911d0bb2-c56e-46bd-af4f-8d9611183bb7")
)
def make_variables_a3(condition_occurrence, drug_exposure, measurement, person, visit_occurrence, procedure_occurrence, observation, concept_set_members, death , concept, COVID_Patient_Summary_Table, compute_inclusion, my_codesets):

    cohort = compute_inclusion
    cohort = cohort.selectExpr("cohort_person_id", "person_id_orig", "index_date index_date", "age", "cohort_end_date")

    # Use our version of the codesets
    concept_set_members = my_codesets
    left_join_cte = cohort

    #
    # Slice 3
    #
    # 75:glp1_before_365
    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "167455542"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -365) AND (datediff(drug_exposure_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    glp1_before_365_cte = cte_1.selectExpr("cohort_person_id","var_val AS glp1_before_365" ).where("rn == 1")
    left_join_cte = left_join_cte.join(glp1_before_365_cte, 'cohort_person_id', how='left')

    # 76:sglt2_before_365
    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "624781538"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -365) AND (datediff(drug_exposure_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    sglt2_before_365_cte = cte_1.selectExpr("cohort_person_id","var_val AS sglt2_before_365" ).where("rn == 1")
    left_join_cte = left_join_cte.join(sglt2_before_365_cte, 'cohort_person_id', how='left')

    # 77:su_before_365
    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "880450058"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -365) AND (datediff(drug_exposure_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    su_before_365_cte = cte_1.selectExpr("cohort_person_id","var_val AS su_before_365" ).where("rn == 1")
    left_join_cte = left_join_cte.join(su_before_365_cte, 'cohort_person_id', how='left')

    # 78:thiazo_before_365
    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "165856330"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -365) AND (datediff(drug_exposure_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    thiazo_before_365_cte = cte_1.selectExpr("cohort_person_id","var_val AS thiazo_before_365" ).where("rn == 1")
    left_join_cte = left_join_cte.join(thiazo_before_365_cte, 'cohort_person_id', how='left')

    # 79:insulin_before_365
    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "91074072"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -365) AND (datediff(drug_exposure_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    insulin_before_365_cte = cte_1.selectExpr("cohort_person_id","var_val AS insulin_before_365" ).where("rn == 1")
    left_join_cte = left_join_cte.join(insulin_before_365_cte, 'cohort_person_id', how='left')

    # 80:other_diabetes_med_before_365
    cte_1 = cohort.alias('cohort').join(drug_exposure.alias('drug_exposure'), cohort.person_id_orig == drug_exposure.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == drug_exposure.drug_concept_id) & (concept_set_members.codeset_id == "912920320"), how="inner").\
    where("((cohort_end_date is null) or (drug_exposure_start_date <= cohort_end_date)) AND (datediff(drug_exposure_start_date, index_date) >= -365) AND (datediff(drug_exposure_start_date, index_date) <= -1)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "drug_exposure_start_date AS var_date",
    "drug_concept_id AS concept_id",
    "CASE WHEN drug_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(drug_exposure_start_date, index_date) ASC) AS rn",
    )
    other_diabetes_med_before_365_cte = cte_1.selectExpr("cohort_person_id","var_val AS other_diabetes_med_before_365" ).where("rn == 1")
    left_join_cte = left_join_cte.join(other_diabetes_med_before_365_cte, 'cohort_person_id', how='left')

    # 81:pasc_date
    cte_1 = cohort.alias('cohort').join(condition_occurrence.alias('condition_occurrence'), cohort.person_id_orig == condition_occurrence.person_id, how='left').\
    join(concept_set_members, (concept_set_members.concept_id == condition_occurrence.condition_concept_id) & (concept_set_members.codeset_id == "708775231"), how="inner").\
    where("((cohort_end_date is null) or (condition_start_date <= cohort_end_date)) AND (datediff(condition_start_date, index_date) >= 0) AND (datediff(condition_start_date, index_date) <= 180)").\
    selectExpr(
    "cohort.cohort_person_id AS cohort_person_id",
    "condition_start_date AS var_date",
    "condition_concept_id AS concept_id",
    "CASE WHEN condition_concept_id is not null THEN 1 ELSE 0 END AS var_val",
    "ROW_NUMBER() OVER (PARTITION BY cohort_person_id ORDER BY datediff(condition_start_date, index_date) ASC) AS rn",
    )
    pasc_date_cte = cte_1.selectExpr("cohort_person_id","var_date AS pasc_date" ).where("rn == 1")
    left_join_cte = left_join_cte.join(pasc_date_cte, 'cohort_person_id', how='left')

    return left_join_cte

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ac828b98-2947-43c3-98f9-f3f0fd9c3ff0"),
    concept=Input(rid="ri.foundry.main.dataset.5cb3c4a3-327a-47bf-a8bf-daf0cafe6772"),
    concept_ancestor=Input(rid="ri.foundry.main.dataset.c5e0521a-147e-4608-b71e-8f53bcdbe03c"),
    concept_relationship=Input(rid="ri.foundry.main.dataset.0469a283-692e-4654-bb2e-26922aff9d71"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6")
)
def my_codesets(concept_set_members, concept, concept_relationship, concept_ancestor):

    # Create additional codesets

    # ckd45
    ckd45 = concept_set_members.filter(F.col("codeset_id").isin([778290097, 742328432]))
    ckd45 = ckd45.selectExpr("'ckd45' as codeset_id", "concept_id", "'ckd45' as concept_set_name", "cast(1 as boolean) as is_most_recent_version", "1 as version", "concept_name", "cast(1 as boolean) as archived")

    # Add Height, Weight, bmi
    height_weight_bmi = concept_set_members.filter(F.col("codeset_id").isin([776390058, 754731201, 65622096]))
    height_weight_bmi = height_weight_bmi.selectExpr("'height_weight_bmi' as codeset_id", "concept_id", "'height_weight_bmi' as concept_set_name", "cast(1 as boolean) as is_most_recent_version", "1 as version", "concept_name", "cast(1 as boolean) as archived")
    
    # Comparator medications : ivermectin, fluvoxamine, montelukast, fluticasone
    comparator_meds_all = concept_set_members.filter(F.col("codeset_id").isin([409971724, 762254277, 858141871, 751977747]))
    comparator_meds_all = comparator_meds_all.selectExpr("'comparator_meds_all' as codeset_id", "concept_id", "'height_weight_bmi' as concept_set_name", "cast(1 as boolean) as is_most_recent_version", "1 as version", "concept_name", "cast(1 as boolean) as archived")

    all_cond = my_codesets_icd(concept_set_members, concept, concept_relationship, concept_ancestor)

    steroid_concept_ids = [43619445, 41328167, 41268124, 19032450, 41422953, 41334382, 41355979, 44097108, 36781615, 41122201, 41333017, 41121398, 43031869, 41413008, 41026355, 42731459, 40972009, 43853156, 41331671, 41413010, 41258018, 41407256, 43662099, 41122169, 41091070, 40883717, 42715955, 21075347, 41030382, 19027190, 40725264, 41164800, 19106651, 41332009, 41341082, 41368503, 41277931, 40872532, 21091038, 41355800, 41216055, 19029657, 41402247, 43133393, 43716694, 41184761, 43607798, 43596212, 41351737, 40905638, 40852466, 40884951, 36073747, 41091090, 42922453, 40905631, 43138472, 43752786, 41034515, 43210218, 41329750, 43207274, 40853712, 21099584, 40995323, 41274533, 41021314, 40959113, 41218175, 41329955, 40968025, 43625913, 43662217, 40995328, 43584422, 41028822, 41339561, 41333249, 41023968, 36891844, 40901668, 41334694, 920626, 41352972, 41271182, 43654979, 21040712, 40899020, 35787680, 44025423, 41052541, 41242609, 41345504, 41337984, 43828005, 40835874, 41338833, 41303658, 43275225, 41039429, 43583425, 41358198, 43716775, 42934714, 41335791, 41274522, 41241039, 43582413, 35777185, 36265826, 40729776, 21099588, 40738362, 36895181, 42934751, 41355471, 41331510, 40878711, 41028281, 40932756, 43829332, 41336868, 41339699, 41397192, 40947246, 43157303, 21070251, 42939681, 41332332, 41332137, 43169801, 40930340, 41340308, 40833872, 43709699, 40824598, 41329058, 40927822, 41339562, 41412894, 43860089, 36063369, 19067792, 41401789, 42942784, 41215293, 41388809, 41216059, 43650407, 40865633, 40866348, 43762314, 40843458, 35763126, 40171343, 43155455, 41272495, 43759516, 36270913, 43799626, 41178641, 41065633, 19080702, 41329622, 40897485, 41337051, 40019099, 43175358, 43146011, 42941645, 42922390, 41343820, 41032084, 43646524, 40717799, 41209377, 41336718, 41330123, 41355886, 43788579, 43715873, 41353714, 2055934, 42934861, 40992699, 35763618, 2057289, 40934753, 41191073, 41407200, 40853710, 41228463, 2057277, 41355678, 36278039, 40866346, 40018867, 43862927, 41094862, 43721392, 41336348, 41338369, 43269803, 43787647, 43800293, 43834617, 40234052, 44190840, 41159944, 2057262, 43212001, 43219433, 36783373, 43189957, 41388366, 43516097, 41332448, 975168, 21060247, 43258713, 40855733, 2057307, 41258015, 21128907, 41292459, 41270533, 41184775, 43185411, 41028284, 41345796, 43619036, 40964023, 41333361, 41111782, 41245207, 41335106, 40887271, 43691186, 41409559, 41345843, 35779363, 41247086, 2057297, 41321702, 40965880, 41358510, 43168166, 41042728, 702628, 43701694, 41332683, 41330166, 35886833, 19095136, 41247112, 21055588, 21042128, 41333942, 43805708, 41336161, 36054456, 43610990, 41330747, 40717867, 40997251, 41059459, 40732467, 35830395, 41331414, 41344603, 40976991, 41333251, 41337186, 2919318, 41147613, 41103360, 41444842, 43626766, 35851447, 43295296, 40073532, 41313092, 41335784, 41245205, 43190111, 40234040, 41323799, 41196146, 41280001, 35779687, 41206868, 41319368, 43260787, 41091082, 43764026, 41330655, 41334419, 43674231, 43034973, 2034603, 41279979, 41182951, 40997178, 41113541, 43146018, 35786757, 40997219, 43823823, 40883734, 41146907, 35160373, 36063377, 42934847, 40941056, 35135542, 41409957, 41328003, 40962472, 41345041, 41334752, 43196372, 43590338, 41084117, 41030375, 41021332, 21037915, 40968019, 41228474, 40976989, 41309192, 43179244, 43277283, 41303657, 41408041, 35777332, 43674795, 43835839, 43171059, 41277929, 35777405, 36035952, 36279362, 40872524, 41097161, 41330646, 35886746, 40241504, 41331294, 40825153, 41407928, 40947255, 41350696, 35886749, 41335262, 21163689, 2057243, 40870605, 44816140, 40976975, 41052538, 41082304, 41375914, 43665374, 41335751, 21163692, 41425145, 41331296, 19113764, 21089688, 41059452, 41089077, 43841981, 40898456, 42934734, 35762488, 43211985, 43824820, 41284160, 41134797, 41253151, 41039416, 35765910, 40872497, 35784797, 21077938, 41151894, 43806649, 40936914, 40924973, 41315380, 43295294, 40965879, 41180277, 41328612, 41370056, 41146713, 41343469, 43626765, 41346136, 41164808, 43157305, 21158612, 41359682, 43142463, 43592419, 35129970, 41345657, 36407722, 43625911, 41335327, 41336629, 36037493, 41357576, 40749549, 40717160, 36063792, 43726416, 41328035, 43592359, 43218483, 41154502, 43753901, 40823347, 43715757, 40936913, 41344870, 43157306, 43185410, 43291305, 920596, 35770849, 21168402, 41329225, 41328279, 43781656, 41344872, 43035010, 40964036, 19007005, 40987460, 19006966, 41401769, 41357014, 35784751, 43180854, 43662182, 36892759, 44060943, 41330545, 41330991, 41091073, 43699418, 43144259, 21173455, 43201178, 43644957, 41352672, 41084138, 36063367, 43146049, 40055571, 41122174, 41159927, 40085185, 41159921, 21109441, 40122522, 41317500, 41351789, 42934875, 41334700, 41299111, 35144066, 40990354, 19007010, 2034608, 41351198, 44159026, 41237128, 35153870, 41352806, 40085188, 42934721, 40995319, 36407980, 43860079, 41320474, 42873632, 40959115, 41345901, 41355675, 36056985, 35160018, 41341461, 43830186, 21087847, 43715823, 36813955, 44102609, 41216627, 41184760, 41166067, 41052542, 36263123, 41328837, 43168157, 43769841, 21060307, 41353838, 41372957, 40718370, 42481387, 40749479, 41335328, 41115543, 43835838, 43590808, 41259307, 41327708, 43601435, 43680131, 41338495, 43753994, 41023665, 21050438, 41080853, 41355977, 43852572, 41153767, 43845097, 40018860, 41407177, 35886831, 43755043, 44098960, 41115366, 43603417, 41335750, 40978234, 41030374, 41400327, 40749487, 41091056, 41309191, 41330346, 41034505, 41333066, 41330016, 41091042, 36889578, 42948534, 41136908, 41333738, 40872508, 44161819, 35149856, 41277937, 44123137, 43626767, 41367287, 40928076, 41328371, 40749536, 41134792, 40825654, 43168479, 43607784, 41277917, 43609092, 42902440, 43863857, 41091092, 41328014, 43633801, 40878687, 21075359, 41242612, 40874689, 40997225, 41209384, 44064632, 2057379, 41039435, 44036462, 43698042, 40849576, 41341777, 41304576, 40978232, 40073197, 41408261, 41021313, 43860220, 41143709, 41343483, 41337044, 40872481, 41330041, 40716465, 21030811, 41332334, 42934844, 42876517, 40995326, 41333945, 40153792, 41342103, 43717189, 40939538, 43771113, 36887918, 41309165, 43644119, 41328036, 43769902, 41358171, 41357976, 41337258, 36886995, 40936903, 589929, 41416194, 41363596, 40852475, 41356698, 41180290, 40839450, 40878703, 41452213, 41385314, 41364179, 40725706, 35777892, 43663058, 21158593, 41091053, 43666706, 41336468, 41330433, 41340514, 43638304, 43637555, 43752680, 35134538, 41286171, 21021124, 41331962, 21079933, 41147174, 41345412, 36037488, 43734854, 21038537, 43168480, 40872504, 43146012, 41445496, 43727419, 41286181, 41334027, 41032118, 41336425, 41356306, 35761825, 40224101, 41093134, 43618823, 40946036, 41335052, 43861066, 40962477, 41336573, 41352232, 21153808, 40959853, 41216076, 41337751, 44183691, 40872509, 41094839, 41276164, 41376097, 41444849, 41330627, 41072808, 41412442, 40941051, 41061515, 43717229, 43824106, 41352877, 41309167, 44190833, 40718198, 41371091, 41328004, 40847584, 41091058, 43610977, 40901674, 41030393, 42722634, 41336209, 35777248, 40073438, 41339920, 35131515, 41340874, 41345783, 43201188, 40941068, 43663508, 41247100, 40909708, 40725272, 41182950, 41003375, 43607782, 35408275, 21138805, 44032109, 19033622, 21148653, 43607785, 40085516, 2055930, 35134585, 41184748, 43157304, 40992889, 40965889, 41133490, 41341434, 43291304, 35604733, 43733936, 41341839, 21150015, 21061660, 41333252, 41284144, 41216109, 35140199, 40883744, 41268428, 41228485, 41354330, 41413144, 1518751, 43035002, 36884997, 41277955, 42922378, 35142087, 41191102, 41034490, 41327542, 41198301, 43824812, 35745118, 44178789, 21124122, 43188394, 43517332, 41208097, 43807208, 41336892, 41340919, 42723524, 21026165, 43726448, 40934749, 43769760, 41356697, 21048396, 41247139, 43825343, 41227229, 41328033, 21163619, 43170479, 21059044, 41028268, 41330815, 36074599, 41259296, 43601890, 43038702, 41337095, 40225702, 41377500, 35780435, 40932758, 40946037, 43180855, 41153745, 21068974, 43146035, 40725705, 41247141, 21124124, 41407601, 43516926, 43289860, 975588, 41352970, 43639477, 36784241, 41039427, 40997216, 41409540, 40917952, 40975828, 41143711, 41327916, 41329327, 41070705, 40947250, 35786209, 41337091, 41337082, 40934705, 35828577, 43035011, 21022444, 41330354, 41053292, 41346421, 41456230, 41128500, 41341910, 41253145, 41284128, 21060294, 43699419, 40997230, 43798360, 41361937, 43637554, 2031945, 40941054, 43663548, 41271852, 41334091, 42942767, 41149170, 41451325, 41408262, 36895218, 2057375, 41309187, 41329500, 21097582, 41330789, 41249185, 21167584, 2034623, 41352394, 21107007, 41355569, 41020696, 41330653, 40870600, 21138824, 43751838, 43601714, 35779282, 43701739, 44034850, 41309198, 41337662, 40228301, 21140171, 43211784, 44064644, 43169806, 44190276, 41329224, 19034012, 41333386, 42922444, 35150868, 41307265, 41352179, 41338533, 40836820, 43727431, 41268421, 41216058, 41356923, 43148942, 41128515, 41330397, 41341394, 43618825, 44173777, 21176138, 36781657, 41122172, 42876077, 41065619, 43700813, 43771278, 42922448, 41184774, 40839466, 40941050, 2055933, 40899012, 41353771, 40853726, 41309188, 41330018, 41413228, 43135166, 43146041, 41351126, 41328775, 21049667, 43733862, 41329320, 19007004, 41329322, 35129197, 41193137, 41332164, 40738869, 35750569, 41184749, 43726991, 43133399, 21077665, 42922451, 43589940, 44182291, 44034262, 41328280, 41413229, 41401704, 41289150, 41289144, 41105495, 41332557, 41445921, 41328218, 41445550, 2057365, 43771929, 43592085, 44045608, 40965923, 35156436, 41103356, 41247135, 43157576, 43709045, 40865008, 43690612, 41073235, 43674229, 41309186, 35765908, 21168401, 41340407, 43625900, 35757659, 21070271, 43800056, 41334156, 41369918, 42918693, 41052536, 41115344, 41184756, 41166053, 42481542, 41327711, 41336633, 42921561, 44114044, 36895220, 41122223, 36882878, 41332385, 36267163, 40843462, 40867945, 41094860, 41408873, 41093128, 41402333, 41122170, 21104730, 41332346, 43680182, 43293416, 41329814, 21175744, 35885558, 40738853, 41332441, 41122195, 41337749, 43201184, 41120317, 43716692, 41363910, 35757657, 41335778, 44084339, 41245191, 41335742, 40845203, 41345197, 41327640, 41214252, 42876080, 41097167, 43289034, 44128589, 41008356, 43773710, 1551004, 21099579, 43142529, 41242611, 19032449, 40927830, 43683700, 40930675, 43625767, 41059465, 41216083, 40837889, 41153772, 41184751, 41329956, 40054910, 43787850, 41334983, 40941083, 40073196, 40927814, 40905136, 40965862, 43860226, 35762069, 42719972, 41354428, 43753091, 35412170, 41216079, 40965855, 40717800, 41018422, 40928512, 40853723, 40907302, 40938669, 43711410, 35604771, 21094863, 41114698, 40974040, 41086458, 21070259, 41333344, 41209378, 41341693, 41344652, 41262273, 41104285, 41330353, 41357615, 41105492, 40841321, 40934745, 36891619, 40085519, 43788584, 43680174, 41026356, 43691875, 36893741, 21030815, 40872523, 40073439, 41177863, 41340776, 41340270, 41329529, 44127625, 41273799, 41053299, 41026360, 43805692, 40941071, 41207352, 43673124, 43715826, 43042475, 40934736, 1506312, 36780904, 1550572, 43823822, 43196922, 44192232, 41334837, 41413192, 41274529, 21138813, 41329006, 41351930, 41040658, 42941669, 41355098, 41276167, 40874688, 35886747, 42934745, 43656423, 41342485, 41218174, 40987158, 40909684, 40941080, 35133956, 41177216, 19127774, 43028405, 36814334, 40928505, 41335780, 21079170, 40941093, 42948533, 36880018, 43042583, 19070310, 41153720, 44126658, 2034700, 42922389, 40878693, 2057256, 40839465, 43163939, 41332395, 21138817, 43166409, 40749560, 41359801, 41091038, 42720955, 36784707, 41335955, 41271851, 40929193, 2042096, 44101650, 40749440, 41091079, 21137953, 41330384, 41335585, 44025239, 41422954, 40866952, 43180856, 920597, 41335752, 41122527, 41412349, 36063781, 41328172, 36063384, 41359383, 40841308, 41351758, 40843453, 36058283, 43646847, 41346362, 36272261, 37499309, 41061507, 43197640, 41071921, 43625866, 43283586, 41451315, 41332554, 35771554, 41122183, 21159968, 41227219, 36036037, 43764030, 41424801, 43751747, 21148655, 43028398, 2057343, 43040543, 44130389, 43201490, 42922455, 41327740, 41329271, 21117811, 40832214, 41337144, 41179694, 21168390, 40914668, 40927832, 40914657, 40738354, 41332446, 21070262, 41124309, 41307237, 40946032, 40999366, 41053295, 41346035, 41320473, 41332388, 43852933, 41336476, 41331874, 41335967, 43269589, 40941031, 41335326, 43196610, 41335086, 21116400, 40718927, 1971417, 41416641, 41329553, 40965918, 40837890, 41337546, 41166961, 41216085, 43590125, 1550975, 35139739, 40959112, 40839463, 43853559, 44190761, 41122197, 41336491, 21109436, 41355318, 21028109, 41284145, 41214265, 40905633, 41271855, 43637800, 41332138, 35777639, 41332660, 44135656, 36054465, 42934846, 41337369, 35745124, 41145118, 35754167, 21042129, 41321708, 43618519, 43770952, 44199173, 41023667, 41330165, 41350681, 41330643, 40872493, 40027453, 41352162, 36277985, 41177879, 21127009, 41022029, 43648160, 43144260, 40884945, 41284130, 41409190, 41279989, 43762683, 21155688, 40991075, 21091043, 43255010, 41334159, 40909691, 42922384, 40716271, 41218194, 36057005, 21087012, 40878696, 41459053, 41396389, 42712969, 43291357, 41197431, 40866948, 43177442, 40749438, 41366745, 41336820, 43210217, 41356596, 41214268, 43751833, 43197554, 35757654, 40932770, 40018866, 41376107, 43135151, 43798408, 41081187, 43715762, 43637548, 2012260, 41210049, 41115649, 43805693, 43190130, 40234051, 40749971, 21077153, 41338428, 43715736, 35865664, 40841329, 43800679, 36063784, 44072508, 41028254, 40729815, 43715758, 43518148, 41333571, 43152910, 40905623, 43636294, 41219859, 41336429, 36783372, 40836822, 43805713, 43649734, 21028108, 43745408, 21067272, 40244441, 43039527, 21070245, 41444854, 1832194, 41336475, 43607878, 43734858, 40927829, 41217726, 40715936, 41284152, 21057416, 35155833, 41331299, 21099520, 43286023, 41289147, 41473738, 43258768, 41271860, 35851449, 43645027, 40841349, 40934757, 41385609, 41122190, 41421396, 41224136, 41331484, 35130902, 41343198, 43625872, 41102122, 41352177, 41352037, 44202578, 44074764, 21096664, 41332129, 41368501, 43625766, 41331980, 43655640, 43764658, 41034519, 41335776, 41071931, 40228391, 41337744, 41258029, 41336208, 21051862, 41344871, 37593842, 42723584, 40878694, 43769838, 41328813, 1972855, 41311329, 41377662, 43673760, 43157283, 36419764, 35777014, 35765915, 43219127, 40903509, 19018906, 40914660, 43763333, 36809978, 21040705, 41030391, 41364423, 36892851, 41159931, 41329737, 41198716, 41084854, 43042595, 41334158, 41421563, 44037854, 43039531, 42939640, 41178638, 41337980, 21119118, 35753477, 41260196, 35866487, 44191558, 41380948, 41049716, 43755442, 43680158, 41341243, 43697929, 43680043, 1518872, 41245188, 41356700, 41039422, 41338991, 41331322, 43201201, 21157795, 40059478, 21097404, 41365034, 40865010, 44042956, 41328238, 41333497, 41381566, 40934752, 41191527, 35866482, 41330017, 40975825, 40934708, 920665, 41277911, 41355345, 21070241, 41115361, 44197719, 41180286, 41354011, 41341289, 35135852, 2054957, 36895048, 43625764, 41022032, 41397942, 41358193, 41388052, 43644117, 41329269, 41177876, 21119133, 43751869, 41327748, 43825348, 43662395, 43039528, 41240619, 43625773, 41356598, 40896206, 43199415, 43716070, 41337048, 41343721, 36781268, 43168162, 41330984, 41380904, 43860241, 40135413, 43584787, 44162087, 41336056, 41247656, 41332134, 41409724, 43805773, 43843096, 44199232, 41302313, 41330019, 40977000, 40949302, 43726988, 21138827, 41053297, 42970588, 40894287, 41001030, 19129290, 41357975, 2057279, 21124131, 43157289, 41458095, 41354910, 41408943, 40127030, 40927833, 41115370, 44161042, 41358602, 41117729, 41331204, 43034966, 37593924, 36277694, 43842886, 41339918, 40878686, 41444841, 43717227, 41145607, 44202657, 2057296, 41331701, 40997246, 41253165, 41369723, 44178788, 21047745, 44062056, 43671911, 43644039, 43256485, 35606542, 21077666, 40915852, 41224164, 2057244, 41193142, 36063361, 41333255, 41100983, 41382326, 41363502, 41317496, 41331713, 41397607, 41343784, 41009586, 43813148, 41357322, 40010223, 43589845, 41336073, 41284137, 41343532, 43042473, 21088478, 41337709, 41330651, 40846123, 41335836, 41339371, 40049701, 41456329, 19098154, 21158589, 43827617, 41329920, 40941075, 41382269, 43526140, 36891843, 41445799, 21117187, 41277925, 41373976, 43673767, 42724566, 35751047, 41099184, 35754794, 21077648, 41336347, 41451328, 21028112, 41057536, 40883740, 41307251, 41330821, 42729021, 43517841, 920918, 41269947, 41338009, 41159934, 41113540, 36279184, 21106522, 40901670, 36056900, 35757653, 35135417, 40949295, 41376100, 44045301, 41089081, 41034483, 21104734, 35784818, 40924090, 43275174, 41329915, 43823941, 41247127, 41336001, 41416384, 41284148, 43782444, 21040707, 36037452, 35771558, 36056999, 41353899, 41222206, 40969685, 44165137, 43700613, 21158605, 42873630, 41222217, 41177869, 920625, 41353272, 36881339, 40991741, 43636925, 41352807, 40989090, 35786755, 43675687, 40958455, 41342559, 36508682, 40980295, 40073456, 40932755, 40900086, 35761840, 43691571, 36063803, 40946041, 41336470, 41366050, 41414135, 41182941, 43662220, 40980293, 41328309, 35138506, 41329352, 41357835, 44190405, 21159971, 41321701, 41354648, 43267296, 41327915, 43141514, 41311341, 41070713, 35753475, 41370191, 41261487, 21150012, 41039428, 21106526, 41327709, 41377562, 41214259, 41329013, 41099200, 40926037, 2054961, 35749549, 41380516, 41082303, 36056979, 41355344, 41393390, 43823812, 41332869, 40934750, 40717893, 43769846, 35886832, 21168410, 41224147, 41095741, 36063366, 43582421, 41105500, 43736120, 43644024, 41332169, 41330432, 44125175, 36036035, 43718197, 41309141, 41353317, 41191096, 41330819, 44128590, 40852487, 40997235, 42922456, 41344534, 41413142, 41330988, 2057363, 36074597, 43269590, 41354459, 43809572, 41330128, 42934750, 41337773, 40934747, 35865017, 35757913, 41028250, 41153766, 41271185, 2012273, 40999356, 40878702, 41304574, 40926038, 41408616, 40839464, 21125997, 41331871, 43518043, 41354092, 21045808, 41328816, 41425057, 41358330, 41331509, 41146309, 44074760, 40839468, 41008342, 40725276, 35758906, 44126654, 40917174, 21077149, 41120303, 41366251, 21028709, 41334224, 41151887, 36054463, 41341653, 41355470, 41329222, 44073343, 35884672, 21176167, 40884958, 36037487, 44175033, 43201189, 19034205, 43146019, 41292458, 41279988, 41335093, 21158588, 41084486, 41331042, 41151886, 41336206, 41342515, 40884943, 42934742, 41385717, 40958447, 43698958, 44116094, 40872505, 41385610, 21148659, 41357145, 41320467, 40968028, 1507707, 41340450, 43179230, 2057298, 41335405, 40976971, 41083495, 43035001, 40749431, 43764308, 43799306, 44063074, 44025735, 43708482, 2057308, 41237748, 41336634, 40026668, 43798362, 35885560, 35761830, 41359076, 40903543, 40831255, 43637541, 43860178, 36063779, 36418763, 40018864, 41249181, 41020689, 41328414, 40862057, 41305669, 1592257, 41270551, 21050464, 42942760, 40903540, 2910251, 41335502, 790238, 41089074, 41334982, 40941077, 40995312, 41357283, 40749447, 41120308, 41341346, 41216105, 43698960, 41327668, 43855440, 35767222, 41337741, 40749444, 41328011, 21039354, 40718378, 41099197, 36054453, 2024767, 43824108, 40841316, 41240339, 2057241, 40878947, 41010826, 36812775, 40725716, 41055986, 44073109, 43775418, 41343072, 40927196, 41413040, 40235163, 35780845, 35787494, 36887968, 43751888, 19104123, 41005427, 41216101, 41146312, 41331200, 41354307, 41418275, 41357775, 43764024, 40711224, 36053952, 41361060, 43585302, 42717862, 40865636, 1518293, 40085505, 43842032, 43296712, 41331986, 41070701, 41052531, 40934725, 44070224, 41330378, 43258769, 40965888, 41340140, 41334342, 41333082, 41228475, 41030400, 43168160, 43144267, 43698950, 42718645, 43690616, 21079932, 40824867, 41358915, 1972874, 43148944, 43769853, 41293442, 40932775, 41337087, 41337050, 41352454, 36278782, 43715753, 41332403, 41166070, 41008336, 43273659, 44184563, 21156724, 21070280, 41338627, 41008355, 41018420, 41337374, 41359988, 21146273, 43673127, 40738831, 41328667, 41360199, 42948512, 44172485, 41329000, 41340390, 41153741, 41337802, 43164635, 41103352, 40893654, 41399277, 41301626, 35749965, 41003402, 43769772, 40749452, 43853169, 35142419, 41134793, 41338297, 40872482, 41008329, 41331981, 44036920, 41242614, 41227234, 40896811, 41334410, 42727278, 41343383, 43709778, 43715519, 43679830, 43752695, 35776180, 41302308, 41229380, 19010639, 41155871, 41352925, 40823627, 41339563, 41084130, 41333569, 41388640, 41336466, 35828576, 35152433, 40976998, 41336417, 35851465, 43861059, 41335782, 43835582, 41333736, 41344230, 40934754, 43816772, 41327718, 41336623, 41284140, 35774496, 41339895, 41097133, 21158591, 41451318, 40959123, 43284461, 43284468, 40010224, 41332680, 41214272, 2057386, 44164808, 41164802, 43745692, 40976976, 43152869, 40941086, 43266212, 43133392, 41214263, 41057537, 41241652, 41340460, 43175357, 41334033, 40738332, 41166065, 21075353, 44112837, 41330743, 43664287, 41372339, 44043762, 37497612, 19034203, 40975829, 41359803, 41333446, 43257281, 43823907, 40843463, 21138776, 43709048, 964311, 41335777, 35416659, 40234804, 43773535, 43168159, 40849578, 35776735, 40749480, 21067264, 40738323, 43166408, 41222201, 40972024, 41329763, 41330313, 43647202, 41197415, 35778668, 35777384, 44178579, 35829130, 40924977, 21065353, 41412657, 40903500, 41271191, 41444851, 41259302, 41258021, 19112080, 41091055, 40884960, 41330641, 41022031, 41328168, 40934732, 43733964, 41363501, 21070266, 43031870, 43736004, 41065631, 41320469, 21030045, 19074185, 35830464, 41351127, 41057529, 41240337, 40122518, 41214251, 43582418, 40961466, 40837105, 41371124, 42922445, 43141787, 41365490, 41335325, 43680181, 21156215, 40990625, 41338851, 41359460, 41334891, 41338007, 36895420, 36268747, 21148649, 41153747, 2039827, 21158599, 41003367, 19106690, 19034327, 40845201, 41053290, 41040660, 40841302, 43211787, 41327434, 40738353, 41351256, 42939631, 40729838, 41342568, 41328833, 21127919, 41331046, 42941643, 41147077, 35416657, 44060939, 41136910, 41242604, 40234042, 40914648, 43663061, 43844162, 40073459, 43207517, 42948542, 41337701, 41058985, 2039856, 41274530, 40965913, 2031879, 40978235, 41332886, 40947261, 41328685, 41218188, 41181315, 36031142, 40934751, 41178062, 40749449, 21094849, 43152653, 41061083, 41023670, 41387054, 41228478, 41239685, 41034491, 43773636, 2030887, 43608761, 40930146, 40997223, 41335382, 41332871, 21021105, 41327513, 41333734, 21099585, 42934852, 40995310, 41328010, 35777564, 40908296, 2057380, 41340840, 36054449, 41331669, 44191523, 36882201, 41008348, 43680965, 41036495, 21032181, 40852490, 41407434, 40834504, 43035009, 42939684, 1719010, 41228471, 41338013, 36267115, 40914650, 43146037, 21109429, 43163788, 41080242, 41159926, 41354329, 41216099, 41335561, 21138804, 41425938, 43197229, 36037297, 36055496, 41333080, 43636287, 43211979, 40883720, 43625790, 41372337, 21099605, 35885741, 43685186, 40987795, 43752787, 41335096, 21168420, 36063786, 42922383, 36031143, 36892902, 41155864, 41336741, 40725713, 41364747, 41339578, 41334870, 41357243, 41358884, 21119122, 41080854, 43769771, 41335739, 43769840, 40872502, 44192121, 36073746, 43618484, 40872478, 40234048, 43836060, 21099595, 43860393, 41230589, 40073455, 40856718, 41199515, 1830974, 40932754, 21040700, 41055985, 43698075, 41331984, 35742715, 41350746, 2057341, 21089673, 43726449, 42479777, 40999371, 41228477, 43664234, 41359947, 41299112, 43589895, 41328376, 21026170, 21158608, 41039431, 41084140, 40738350, 41415584, 41366051, 41151874, 43146050, 40825863, 40941336, 41153755, 1592256, 43207723, 43646125, 40976997, 41445733, 43729345, 35742713, 43135172, 43717178, 21060305, 35767213, 41331452, 43291834, 41028245, 2057288, 21109446, 41332612, 41330820, 43751840, 44122789, 40905630, 41071920, 43273664, 41059483, 41307254, 41320476, 40018168, 35778809, 41018427, 35149915, 1719045, 43781405, 40839477, 41334427, 41334028, 42479776, 19104491, 36278181, 43860090, 40997194, 21158610, 41323803, 43031874, 40171336, 41392580, 41332339, 40941069, 41009597, 21086901, 41230588, 43734860, 35129926, 40843479, 41345151, 41309178, 41330349, 43806654, 41151873, 41342324, 40987453, 40716042, 36055509, 35774483, 41115360, 44095433, 21150008, 41256883, 41161936, 40718371, 41331295, 40903512, 41291676, 21094847, 44072825, 19080181, 41338011, 41177868, 41320485, 43618527, 42479773, 43701235, 43212011, 41329626, 41334872, 41309701, 43157328, 41041930, 41335265, 41239696, 41158429, 40073195, 43657008, 40927816, 40150051, 41311328, 43196565, 21040715, 41177851, 41332776, 44185762, 41330168, 41122184, 41341861, 35865667, 36258419, 43286024, 43289859, 41385817, 41309700, 41332658, 40966267, 41331482, 41164803, 41302304, 40833868, 41344811, 41199513, 41337264, 2031947, 35830463, 1506314, 43177439, 42948517, 21060315, 40717878, 43691813, 40927194, 41362576, 41168093, 41132328, 41184757, 41184791, 41299423, 41407844, 41417255, 41334693, 35135630, 41328410, 41409936, 41159946, 41451332, 41333939, 35865031, 43662198, 40749553, 43610024, 43179231, 43691588, 41379005, 41253143, 41360126, 40947251, 41309168, 40025524, 41354723, 43179535, 21097387, 35776902, 43288537, 43764027, 41329323, 41334749, 41328424, 2057305, 1972877, 43798369, 43201485, 41332383, 43745694, 41328007, 41333240, 42876076, 21168394, 41039420, 40234803, 41337752, 41412533, 43612410, 41146300, 42934731, 40830330, 40896820, 2034625, 41122191, 41258019, 21153754, 590101, 40843456, 40732456, 41237127, 41017247, 41385819, 21138797, 43811044, 44072828, 35134058, 40995321, 43780419, 35159141, 41330649, 41227230, 21119130, 43829268, 40949299, 43199416, 41357834, 41100981, 40843460, 40749472, 41354142, 21077157, 36063382, 43669861, 43190430, 40839476, 41059499, 41247107, 41350618, 41057538, 40825220, 41153705, 41332555, 41387053, 43662177, 21100970, 43853566, 41159953, 41051900, 42942763, 41328817, 40964021, 42480361, 41323800, 21070261, 41352453, 41333769, 41279980, 41445638, 40852480, 35830392, 41070709, 40729276, 40949303, 40731079, 40028561, 41102140, 41026362, 41342013, 41157554, 40948533, 40749554, 41134796, 41216086, 41214271, 41036480, 41335963, 2054956, 41329735, 2024032, 44075329, 40866344, 44197706, 41354041, 21155814, 41445734, 35159659, 41339822, 43644157, 40915866, 44136434, 40949301, 41289138, 43158933, 43715821, 41332609, 41277922, 36057002, 21138740, 41339700, 21153753, 40903546, 41328040, 43207465, 41355570, 44190262, 2042115, 42934739, 2012271, 21165505, 41413143, 43860229, 40729274, 36887714, 41186917, 41339504, 1592253, 41102133, 1510438, 42942761, 35780547, 41309157, 40999357, 41337136, 35886743, 41034512, 43608759, 21176250, 40936912, 43841909, 40978240, 41353345, 41376106, 43672489, 41258017, 43271284, 41230587, 41009592, 41184773, 41330167, 41216056, 41074060, 21173457, 41103358, 43691575, 42934873, 41057548, 41247123, 41351683, 19129883, 35758902, 43626101, 41362707, 41332844, 41456093, 43135449, 41327706, 40968031, 40841343, 43647746, 43146013, 35746935, 35766391, 41128512, 35886745, 1972355, 1832054, 2057287, 41309197, 40978248, 21059259, 41353506, 41351931, 41334424, 42953130, 2057370, 41334052, 43699441, 41323798, 41274523, 41336530, 43034971, 41279978, 41445639, 40992696, 35745116, 43763472, 41153751, 19098197, 41159938, 41247134, 21100966, 43701836, 35746404, 41270536, 41329919, 36780570, 40917954, 41332613, 35786242, 43812170, 41255151, 41222214, 40958454, 41271854, 41355001, 41356603, 40841307, 41084126, 41329002, 42922386, 36026950, 41357709, 41315392, 41091050, 41333448, 41346115, 42724843, 40749435, 40976994, 41153752, 41354044, 41335786, 43186594, 41472950, 41309169, 41099203, 19026064, 35779378, 21168398, 40234793, 41327707, 41323010, 35142689, 41091044, 43211991, 40961465, 41033032, 44191535, 41128497, 2012262, 41361938, 41327566, 42941667, 35771560, 41412734, 43667254, 41240507, 43698068, 43808623, 41161951, 40854568, 43735219, 35774488, 41059458, 21158622, 35776553, 21079957, 43690958, 1832147, 41360128, 41008335, 44050425, 35140581, 41439245, 44174816, 41287943, 41249199, 41270547, 41150243, 43751839, 21109442, 41409721, 41444853, 41332552, 21071600, 41333253, 19095151, 40070931, 41339612, 43211980, 40717801, 41358360, 35416661, 40153448, 36262443, 36809707, 42717199, 43719383, 43844692, 41335569, 43201208, 41426221, 41412908, 35146279, 41253149, 42720538, 41003398, 40872491, 41344286, 41153765, 21158600, 41082302, 36073748, 43673355, 40716047, 43823936, 44090273, 44183690, 41155853, 41052849, 43780420, 41336533, 41082301, 36055494, 41315407, 35780738, 43201481, 43626770, 43618824, 40903495, 40874703, 41329908, 41050352, 21032187, 41358366, 41342179, 41159919, 35139156, 35604744, 40843471, 41339374, 43806984, 43619633, 40915860, 43852574, 36035955, 19133985, 41153703, 21153875, 41080855, 590256, 43024878, 41247124, 41329353, 41357091, 41307239, 36279254, 41385753, 41337376, 41248731, 41328614, 43701238, 41091064, 41290393, 41070719, 41336204, 40109740, 43135141, 41093136, 40863090, 2057282, 41050049, 40729789, 40872489, 21143859, 42715165, 40872538, 41444843, 41247099, 41105497, 43628444, 40823035, 41337904, 43644158, 40843466, 41336164, 35759415, 41302319, 44101191, 36886753, 44064643, 43680132, 40959852, 40990358, 40731076, 41337267, 41030386, 43841910, 41337109, 41335741, 41345250, 41149190, 41063240, 40085193, 21166010, 40841298, 36037298, 41117119, 40965874, 41184788, 43824107, 35787650, 40883716, 36509686, 43708828, 40717023, 35886744, 35865663, 2042089, 40841346, 2010005, 41309183, 41334341, 35778553, 43627954, 41352926, 41342547, 41333770, 41341909, 41329452, 43788580, 41381359, 43790194, 2920791, 19063927, 43035006, 40978236, 41342397, 41340642, 41267407, 41146302, 43584782, 41026350, 36891530, 41009596, 41353958, 40825591, 41112424, 41333940, 43589805, 41277962, 43672768, 21135871, 40914647, 35410898, 41402931, 43038555, 35897670, 36063802, 41333077, 41364901, 44089590, 43142532, 40234057, 43034972, 36267164, 43769867, 41330386, 40900083, 41344075, 41336624, 41074062, 1550654, 43207776, 41178636, 43208505, 40903548, 41191094, 35886734, 41329909, 41333672, 40947264, 41352969, 40738320, 41355441, 43625867, 40877249, 36814447, 41345841, 41339166, 41123854, 40749486, 21143921, 43745406, 43842878, 40839472, 41249189, 21057417, 40888043, 41359283, 19067759, 43860076, 21040713, 41409189, 41087464, 21124132, 43763336, 41386669, 40874708, 43197225, 40729269, 43861562, 45774914, 44194947, 40924975, 21138829, 35778213, 40833873, 41337370, 21147271, 41286170, 40976973, 41355003, 41334056, 43787645, 43179228, 41321699, 43698016, 43157579, 21089698, 35761836, 41247110, 43680973, 41337308, 21037786, 41271190, 41091576, 41336869, 35777985, 19034013, 44084202, 41407724, 21126512, 41413092, 41059479, 21078107, 41332849, 41330792, 41345599, 43218841, 43764660, 41337306, 41365349, 41334413, 36063785, 41286180, 43157318, 40927835, 41333065, 40972037, 40849571, 36896126, 41276170, 41072809, 43787734, 41039417, 42482158, 43657010, 21166200, 40997234, 43291306, 2057377, 41245195, 41457582, 36507556, 41328679, 41284157, 35866484, 41344541, 43157320, 40729265, 35765902, 43841911, 41335094, 43662178, 43672773, 40995324, 41337138, 40878690, 41240330, 41345385, 35750125, 41330355, 40989730, 43810365, 41333768, 44191585, 35886830, 40749432, 41346548, 21079968, 43186593, 41012566, 41224142, 41334428, 41336079, 21088302, 41336427, 41331959, 43715884, 41445549, 43179408, 43827616, 43625927, 41353865, 43201491, 41335959, 40738330, 21117062, 41375094, 40934763, 40965865, 41214262, 40884955, 40839474, 41133492, 21036148, 41193139, 41331967, 41113538, 41327756, 41342740, 43166404, 21040694, 40995317, 41388025, 41329845, 21094859, 41330817, 43179237, 43762892, 2034701, 40717870, 21086907, 43710172, 1551170, 40883726, 41333067, 40028260, 40717873, 41354430, 43190116, 43807155, 41277938, 41354205, 41329698, 41359131, 21140170, 41358508, 1551123, 41352620, 40060705, 41445615, 42948555, 43607882, 41337045, 41329740, 43627201, 43626773, 41333529, 41374752, 41018505, 43800680, 40976993, 43601603, 41330552, 43738021, 41097179, 41218191, 41413285, 21126126, 36056995, 42941668, 41354255, 35771107, 41327419, 41473244, 41338249, 43719879, 41401880, 43860223, 41070698, 43709766, 40841309, 41022680, 41091083, 41328413, 35767215, 43135147, 21158603, 43595441, 21099593, 36063797, 41358724, 43806662, 41343581, 44062629, 43763335, 43662230, 41186923, 41021318, 41222175, 41328219, 40989089, 41115348, 41331052, 43135144, 40749457, 40872534, 43845089, 42948509, 41124307, 41331318, 41182949, 40834503, 43715859, 41352973, 43620789, 43860225, 36026949, 40836808, 44087621, 35606538, 43697442, 40749533, 41345040, 41211674, 41331324, 40073454, 41329734, 41337504, 41320484, 35886738, 43708485, 21104740, 41330383, 43787422, 41335404, 40878681, 41337375, 41332286, 41159940, 21040673, 43824819, 41309702, 35746932, 40997244, 41271186, 41346034, 40929574, 43625768, 35148908, 40905635, 40833874, 41396744, 41333527, 41331450, 41334174, 40866347, 41385832, 41357070, 41355346, 43135142, 43769759, 21099597, 41329998, 41026361, 43644022, 40915853, 41230582, 41091046, 41182948, 41081072, 43024885, 43179234, 43662209, 19006967, 41359381, 43295290, 43168182, 41271861, 40997208, 40749969, 43153273, 43034979, 40847592, 41146896, 35886736, 21166260, 41327961, 43141440, 40872849, 41358913, 19129102, 43182052, 41216095, 35757650, 41330625, 41331581, 1507736, 41070716, 41328310, 41245204, 41276173, 41209372, 41091071, 41247116, 44130388, 43589904, 40966454, 43691185, 43745638, 40738850, 43864304, 41445517, 41342684, 41341290, 43166407, 36063783, 41039430, 41021315, 41178063, 40853720, 41335775, 40980301, 41103355, 41030385, 43589901, 41377265, 41196153, 43665878, 41084143, 43754556, 40999363, 35767217, 41247121, 43177426, 41089084, 21040714, 41345744, 35777772, 21128932, 41334431, 36037299, 41087468, 41342328, 41334426, 36891671, 41408797, 41247125, 40976977, 41336721, 43179224, 41351841, 43818166, 41356601, 43764023, 40738839, 41222186, 35779433, 43823808, 19130001, 41472612, 41330644, 40958448, 41331432, 2055912, 41211697, 36055501, 41358411, 43769881, 43825448, 41328405, 43736119, 40934756, 41337094, 42948562, 41245177, 43590809, 43654657, 44111981, 43188405, 43699496, 41177227, 40738336, 41329049, 41061516, 35757655, 43024873, 41342949, 35407802, 40923843, 2030882, 2042094, 43607903, 41370276, 41330352, 41358364, 43637714, 43607796, 41353054, 41277936, 40991079, 43715735, 43698015, 43700817, 43041666, 2012256, 41480710, 40997187, 44192023, 35886729, 41336462, 40866343, 43157292, 42875920, 41042727, 43716688, 41328037, 41395963, 44114863, 1973095, 41328028, 2935779, 43861062, 41166052, 40896201, 41191090, 44131112, 42934725, 35606545, 40878688, 41334748, 35767218, 35886727, 21119136, 41052522, 41128492, 43034985, 35749312, 43201185, 41329451, 40941078, 2055905, 43515084, 41334227, 42934868, 19125245, 41357707, 44088622, 43685251, 41344424, 41103353, 40139039, 41117705, 41331316, 40915856, 41333943, 40866337, 41338735, 41329063, 35776187, 41222216, 43818804, 41322619, 40171266, 41358197, 21086900, 41352085, 41388810, 44198432, 43626764, 41422252, 41335041, 40852471, 40972018, 36780572, 40913573, 44114864, 41216072, 42720539, 43770746, 40896823, 41355316, 41358461, 41356305, 43844726, 43806659, 36074600, 42902115, 42934728, 41218196, 2034612, 35778554, 21021127, 44172131, 41357747, 43031875, 21114517, 41330220, 36063364, 41341840, 21165616, 36780902, 35829567, 43734859, 41269930, 35770233, 41328465, 43518062, 42939643, 21138801, 41103362, 41134791, 21075275, 40909707, 43135158, 41272485, 19135172, 41103359, 41337481, 40905639, 43752698, 41259299, 21021110, 41209360, 40904281, 1518258, 35751049, 40070932, 41330648, 41342104, 43751771, 43185412, 41332136, 41186937, 41327664, 41331203, 21128926, 41208723, 41332406, 43164692, 43789854, 41407723, 35763616, 44114046, 40028264, 40988578, 40738834, 41340219, 43589937, 21057546, 21048340, 43135450, 41336898, 21130232, 21135875, 43805767, 40887054, 43600472, 43698954, 35146310, 19067793, 41340405, 43644025, 41333785, 43590131, 41003374, 41249193, 41340842, 42948513, 21070246, 41222190, 2034616, 43212282, 43626771, 41337353, 41338001, 41327397, 41327911, 41028261, 42934737, 41331869, 41331711, 21060297, 35154390, 36895212, 41021324, 41302318, 40965894, 21065417, 43661858, 44178580, 40934740, 41330013, 43787655, 43861063, 41219860, 43157315, 41345935, 36280083, 41330395, 43726990, 41309154, 42731458, 40870592, 46276004, 40749548, 19074157, 41302311, 36810846, 44190841, 41330350, 41345785, 41331029, 41333246, 43836346, 41219844, 21030826, 41338429, 35131075, 43202833, 43698963, 41104286, 35141116, 44197868, 41409545, 40073200, 21134019, 2057240, 41302298, 43735395, 41273500, 40865638, 1518257, 41338427, 41388590, 40834310, 41453234, 36418178, 42726437, 41249190, 41021328, 1832198, 41333833, 40872506, 42970590, 41309199, 43755000, 41357746, 41320477, 21169772, 41309163, 41290394, 41358192, 21145864, 41330752, 41334379, 41230590, 44100656, 40997212, 40893947, 41337052, 21165992, 43039529, 41337759, 41247122, 40941064, 40927831, 41289137, 44124684, 41334909, 41335100, 40821479, 41343607, 41249173, 40892891, 35866488, 43755945, 41117723, 41327912, 41330399, 42942768, 41049713, 40866345, 44190278, 43744461, 40884949, 40968018, 41097178, 44190718, 43627335, 2034635, 40049692, 21037782, 40749490, 21130243, 40991082, 41240321, 43179241, 42479030, 41042726, 36063388, 41008340, 43752700, 21079973, 35159205, 43798364, 21075239, 43806656, 43168169, 41339564, 40975826, 40936905, 41008332, 41330624, 36056981, 40865006, 43219877, 41407634, 36054461, 41329917, 41413897, 43273663, 44136571, 41209381, 43637549, 41333079, 41093140, 36054455, 41334376, 41409726, 41040659, 41329527, 43038698, 43589898, 21061663, 21077280, 41420748, 43607932, 41059486, 40073458, 41084858, 43673766, 43526141, 41346036, 40877244, 41458486, 43146034, 41332553, 41054921, 41333648, 44199290, 41328041, 41343046, 44136575, 41186941, 41330287, 43672769, 40028600, 41329918, 43744501, 40738855, 41482386, 43764657, 41065608, 21175222, 41085535, 41344873, 43682216, 41336428, 2057300, 41353343, 21021098, 41409891, 41112425, 41335403, 41214257, 35774489, 41351896, 2034699, 43275160, 43654638, 21028732, 41331985, 41355442, 42720956, 43842880, 43710784, 41240336, 35766792, 41394600, 42942781, 40874702, 41342545, 35755309, 35886737, 2034638, 41277920, 41360879, 41354207, 1551098, 36267503, 41338270, 41329751, 21057415, 43789137, 21138814, 41334195, 41136912, 41335661, 40976978, 41346227, 41067729, 40841344, 21165990, 21087348, 21146243, 21027997, 41338051, 41327565, 41345656, 43526136, 41277914, 35776183, 41320472, 44161114, 21094846, 43715818, 21088660, 41122200, 43682127, 43734868, 41151890, 41040668, 41242610, 41218185, 40901669, 43639478, 41238552, 41451324, 43526118, 43600818, 41353927, 44049035, 41211693, 40972017, 43823909, 21119123, 35783600, 41330548, 43651866, 41328311, 41331242, 41051902, 43662210, 35156488, 41332132, 43717035, 2031949, 43805712, 41329699, 36887919, 41354649, 43589804, 40995314, 42922450, 41034518, 43770830, 21040701, 41340406, 41153760, 41331412, 41336236, 2031946, 43770737, 41332393, 41216051, 41332654, 41168094, 41116101, 35753480, 904171, 41353867, 41413158, 35885561, 21045809, 40978242, 43717884, 40226713, 41053180, 41337483, 41340945, 43762323, 21067799, 1550716, 40872531, 40729813, 41340264, 40934721, 43655634, 43770747, 40073203, 41332677, 35777110, 41155859, 41259297, 35746403, 41164810, 35780436, 41337637, 43682760, 21089678, 42934748, 40729270, 41420059, 43610025, 40965893, 21148589, 40019101, 35754793, 41357830, 43655200, 40883733, 40905626, 41332389, 43781406, 35758903, 41084734, 41337084, 41337983, 21168415, 41331872, 2054960, 43135139, 41008346, 40738335, 43526421, 41354928, 40049702, 40872536, 2034633, 41323801, 41331867, 21077832, 41333090, 40915859, 43681465, 43698962, 41354570, 41166957, 40903539, 21173388, 41159925, 19101719, 41332655, 43133394, 43798929, 41352492, 43201482, 44099636, 21030817, 702228, 43853563, 41061519, 35778051, 41386647, 41259309, 44036918, 43644165, 40909719, 41173847, 43684588, 41327750, 36895338, 41178065, 40936917, 21061665, 41374924, 40946028, 43628007, 40884940, 41334753, 41277919, 43726772, 21026180, 43628101, 41273502, 40965903, 41166960, 2031944, 43271753, 41343818, 41330985, 35606546, 41335104, 40843452, 41166063, 21173389, 41136909, 41339081, 43662396, 41153736, 36420562, 21130241, 41304572, 40903043, 41084859, 41330222, 41337261, 43141351, 2031970, 41166061, 19067751, 41335045, 42712813, 41331454, 43798410, 43201206, 41333089, 41333347, 41354650, 41298265, 19107660, 41336917, 43289864, 41336632, 41328034, 41331321, 42922376, 21042131, 43135161, 41191069, 41070697, 40738840, 41328777, 2034636, 41340843, 21032179, 41249176, 41330822, 41331091, 43644114, 41386056, 40846125, 41259301, 21030830, 41022028, 41329756, 43142576, 41061082, 41328814, 43699793, 41089079, 41331485, 35160024, 40872539, 21070242, 41353868, 43787730, 40930152, 41197424, 43627276, 42482501, 41094861, 1150865, 43662397, 40852486, 35885743, 41344073, 43153588, 41365649, 41444844, 41307267, 43211781, 41277953, 21060308, 41335048, 43291390, 41356853, 44071319, 40738343, 41358722, 21050467, 35774481, 40073201, 41330192, 35781047, 36881886, 41209238, 41327959, 40962478, 41351859, 43744500, 40896803, 36073990, 43708488, 40729774, 41453687, 19130166, 41091069, 19007061, 41227240, 41337661, 43589893, 40833875, 43625910, 40909716, 43028399, 36888680, 41332084, 41351754, 19006938, 35775101, 40852485, 41355465, 40167746, 40852489, 43672815, 43769762, 41166069, 43024880, 41097165, 40928514, 41358910, 41335107, 41030387, 42934722, 41070718, 36037354, 41331866, 44058448, 41091086, 40930153, 43655901, 43190434, 41333670, 40846121, 2057318, 41247101, 21021103, 41345074, 41336528, 41338343, 41329922, 41054911, 41134798, 41387517, 40140086, 41328426, 43041792, 41345132, 41376069, 41340917, 41071924, 41355673, 41473074, 43168161, 41197421, 41245187, 41332399, 41304567, 42948508, 40872851, 40981278, 41026372, 41328412, 41337092, 43834629, 41359585, 36885756, 41286167, 41356527, 41176078, 41344891, 43589939, 41342180, 21116756, 41338852, 40943058, 43584789, 41332675, 41249179, 35161274, 43526415, 41093132, 41353119, 42903031, 41302994, 43295291, 36813873, 43736121, 2024766, 40729787, 41196152, 35897680, 41065618, 41407515, 21070252, 41099204, 41335099, 42480924, 35778057, 43218311, 40054909, 41091060, 21138826, 41356696, 43834957, 43626761, 41184782, 789932, 41209363, 43816418, 41095746, 43040542, 41084134, 21120471, 36063787, 40839460, 42941661, 21086904, 43752696, 35147019, 40911822, 41028269, 19029659, 41064165, 41445579, 41270544, 41153748, 40934737, 41335834, 41253152, 40965886, 41460775, 41355005, 41209362, 41284162, 42479772, 41408049, 41258025, 41008350, 41336744, 41333735, 41120299, 41342947, 41335567, 41324810, 41097183, 43789136, 43818807, 41336621, 43182507, 41331699, 41177225, 40732469, 19029909, 41330012, 41358601, 41335613, 44114862, 41309145, 41345384, 41184759, 41102130, 21060300, 41289136, 43805225, 41097184, 19101991, 41026365, 41334021, 36056988, 41256886, 36055500, 42934732, 41333247, 40916776, 41249192, 40896819, 41214260, 44057654, 35779281, 41167355, 41327425, 40883721, 41346038, 43584784, 41336893, 41289149, 41081186, 36895239, 43752691, 41329738, 44178305, 21027992, 35770614, 43844024, 41311339, 41009579, 41218184, 41333016, 2057290, 21128919, 44063073, 41021329, 41334750, 21026104, 41329761, 43024882, 41327716, 41338045, 40085514, 41355096, 44203074, 43727941, 41335148, 41329733, 41320468, 41335090, 41153773, 43672812, 40872492, 43691573, 35129094, 41336238, 40997193, 21060293, 40883742, 21114513, 41309175, 41377680, 40234098, 43788582, 41112522, 40849561, 41222199, 41249182, 41445920, 43789076, 43164634, 41180276, 40853715, 41243102, 43591160, 43842877, 1551093, 41333502, 41327712, 41333022, 36054460, 40853727, 44195946, 21089677, 21061670, 40872480, 41340074, 2057292, 21168396, 41445787, 41249184, 43690351, 40965854, 35745123, 41061522, 41327743, 40718926, 41197430, 35755310, 43024869, 40060704, 35763619, 41333731, 41338923, 41327791, 41284129, 40980291, 41269064, 44063430, 41328611, 42934849, 41184763, 40903490, 41146918, 1506316, 19019892, 40993734, 41332081, 40872499, 35606557, 43202831, 43701837, 42934826, 41346545, 35151203, 41407319, 35786169, 40997210, 41327920, 42934839, 44109972, 35897666, 43174637, 41335542, 43834958, 41353232, 43860239, 43619037, 43152442, 21169776, 40849575, 41021320, 40968027, 36277846, 41334691, 19067188, 21079962, 44099635, 41334712, 41085525, 21045803, 43179246, 21040693, 40936916, 41130592, 41328368, 44074146, 43155442, 41115345, 43656089, 41163655, 41304023, 40865640, 41059454, 42942788, 41345704, 44190150, 43144264, 41359804, 36037301, 41331700, 40841312, 41334508, 41334319, 41329739, 43145831, 40934714, 40874693, 35777013, 41337747, 41222211, 41329809, 40085197, 40872546, 41335046, 43657007, 40738333, 41153749, 42483314, 41370354, 40872470, 40234801, 41375671, 2057374, 2034703, 43637803, 41149188, 21049164, 41342048, 21079931, 43823825, 41357244, 36277753, 21148660, 41175609, 35865674, 40997240, 41332133, 40841348, 43727780, 44034852, 41387866, 43034968, 41065607, 41336058, 41001010, 44088624, 43592418, 40936906, 41279990, 40905621, 1550719, 41331297, 41339809, 41086456, 41451329, 40941090, 41028249, 41247133, 41151885, 41344423, 40847588, 43625864, 41117133, 43177429, 43719569, 40831536, 41341206, 41334196, 43823340, 43190124, 41032117, 41327670, 42934817, 41328468, 41371874, 43770740, 40716471, 41159943, 41355953, 40841347, 21048341, 42724992, 41341937, 40978238, 40846124, 41336484, 35772113, 41412714, 40934759, 35866486, 41216091, 41333771, 40980303, 43645262, 35778172, 44192255, 44112408, 35762315, 41339053, 41097166, 40231785, 40934707, 21061680, 42922446, 43673128, 35142078, 21096670, 41331708, 41136102, 21099573, 35779664, 41309194, 40959850, 40931194, 41358509, 41065613, 41333526, 41328312, 41337270, 41333019, 41328056, 40896207, 43157302, 35138092, 41328687, 21128922, 42934856, 41191084, 43763584, 43188403, 41330250, 43751879, 43582431, 40865635, 43836349, 43218546, 21120469, 2042114, 40976987, 41345382, 41337089, 2042091, 41227233, 21026179, 44176890, 40965873, 41335501, 43817754, 41328832, 35749688, 41397402, 41149182, 41328615, 21067263, 41355227, 43673351, 41102121, 42934845, 44062627, 41000990, 43680967, 41346728, 41328417, 43517898, 41337485, 41345712, 43582775, 40927193, 21065416, 41102127, 41247137, 41342769, 41134802, 21104741, 41146911, 43637871, 41059502, 41354093, 43157578, 41153756, 43787970, 43824821, 40899013, 40847605, 41340037, 41059985, 43199418, 41198713, 40930147, 19067760, 41039433, 41395962, 36055492, 41284165, 41115368, 41191071, 35154938, 43211988, 41361109, 41317518, 2057237, 41331664, 43682214, 41023669, 43702561, 41330979, 42948558, 43196921, 43818805, 43672764, 21089689, 41331198, 35766775, 35767212, 41355065, 2057261, 41128526, 40749561, 2057301, 2031941, 43729344, 41333015, 43280551, 41229774, 41343088, 41331317, 42726172, 21140178, 41091087, 41191076, 19061517, 43155449, 41409608, 21079961, 40999367, 21168339, 41330788, 1832195, 40968037, 41364569, 41334383, 41328031, 35776267, 43034983, 43654348, 36418179, 41245201, 36810960, 41334026, 21158606, 41020695, 43673354, 41034493, 43293014, 41339503, 35865673, 40931202, 41089078, 41333339, 41115650, 41091065, 43526142, 40965864, 21138810, 36418761, 35765904, 41409686, 41327567, 19106204, 41054924, 41330130, 41337049, 43607880, 41307262, 40938636, 19054212, 40872473, 41388762, 43526117, 41034489, 41414676, 41249195, 41337004, 41154269, 40729783, 21106626, 21042124, 43185577, 41333083, 41057547, 41155862, 41336424, 40934712, 40849580, 43708468, 36054468, 40716467, 41210056, 41199514, 21085071, 35778808, 35143825, 35777246, 44108914, 43591219, 43286027, 43836061, 40897488, 44031344, 36057001, 19076145, 41177853, 41216048, 43764028, 41153730, 41453431, 40872479, 36056982, 43042471, 41038257, 41330995, 40865013, 2055004, 43601430, 40974048, 40883725, 41126079, 21079955, 40928510, 19034204, 41184728, 2030892, 43680963, 41378594, 43188404, 35757652, 41336095, 40718163, 41317513, 40749557, 42483090, 21128917, 40716470, 41346135, 43844787, 40992133, 41331961, 21040706, 41418402, 35141077, 41332391, 41337746, 41249191, 40831538, 43763328, 41330221, 41059474, 41191100, 41261305, 41054925, 41143997, 21070253, 42934730, 41321706, 43772324, 41334378, 41153723, 41332191, 41276150, 41258022, 41344500, 43179532, 41407199, 44096033, 41372525, 21040708, 42479558, 43860224, 41335783, 36420019, 40821900, 41111375, 40991083, 42620788, 43861275, 40917950, 41374700, 41327540, 21155690, 42902107, 41042724, 40738852, 41175022, 35787691, 41227226, 41328009, 1551051, 41328999, 40965922, 41333782, 41133489, 35746934, 43654977, 41333501, 35779283, 41343718, 43618517, 41335191, 43805212, 44192402, 40725277, 43601428, 43736118, 43715755, 21081305, 40869014, 21021096, 41350751, 41091075, 41336162, 41330197, 43289863, 41042723, 43034999, 41197414, 41360131, 21167582, 36894465, 43862243, 41181317, 44076708, 21109444, 41277930, 43855493, 41097605, 41337745, 40896204, 43754554, 36054469, 43200998, 43662173, 41337255, 41337263, 19128781, 43135451, 40823974, 43788585, 21030823, 40959118, 44060036, 41333783, 41258014, 40234820, 40962483, 43700134, 42934865, 41155855, 40990355, 44108284, 41362783, 21021125, 41315413, 41052540, 41186932, 43842882, 41270550, 44109973, 41385265, 43271708, 41330005, 43608812, 40903541, 41304890, 920757, 43835900, 41331607, 36278250, 41343822, 43041791, 41099201, 36886722, 40073461, 41335084, 42722633, 41358796, 19017169, 43710171, 40729778, 41331092, 21037920, 40847580, 2057263, 41091033, 41146924, 44070226, 43762682, 41328582, 40999358, 41336897, 40896809, 41344422, 36277982, 41208724, 41356304, 41284163, 41009593, 41328834, 41328681, 40738319, 41333342, 41335150, 41339051, 40995299, 41337256, 35778348, 41374357, 43211997, 41328057, 35769727, 41344072, 21128913, 43289035, 40749461, 41084141, 40979101, 40992695, 21136397, 40865012, 43715752, 40832391, 40904282, 43781654, 36063789, 41023659, 40738864, 43601721, 43690658, 41334635, 35777247, 43517011, 42922385, 41277934, 19129514, 36063372, 2030654, 43734055, 21030816, 40841326, 43595538, 35138307, 41333647, 41115349, 41444855, 36063368, 43800057, 41059478, 1507738, 35776448, 41341776, 41249175, 40897490, 920946, 41024728, 41329345, 41368146, 41071928, 41335961, 41382056, 21060299, 41353344, 41328029, 21050475, 41028252, 41210050, 41417256, 41315394, 41180280, 35153302, 41005444, 2034694, 41209382, 41346177, 21075274, 41342327, 43682126, 40956530, 41333085, 41102132, 40872525, 19130171, 40997224, 41341460, 40988665, 1719022, 1972354, 35774482, 43766111, 41337265, 21143923, 43834956, 40835201, 41146306, 21135867, 41327669, 21168387, 43601797, 43861615, 36055778, 42483091, 41003376, 41061514, 40936909, 21089690, 41216066, 44197985, 43715822, 41355997, 43262688, 41227227, 41340946, 43144271, 21150017, 43708484, 41329695, 41159932, 41240328, 41330629, 21114518, 35786663, 41331034, 41320480, 42948544, 40974063, 43841700, 2034629, 40909697, 36895134, 41277910, 35745119, 41333730, 35749308, 41302310, 2012257, 21068127, 43142464, 40853719, 41345902, 42948551, 43769842, 40964037, 41332338, 41335753, 41216089, 36781248, 40026665, 43808532, 43710175, 41146310, 44050426, 41174673, 2034630, 41360127, 43816419, 40026666, 41164807, 40901658, 41148213, 36780398, 41161930, 43657006, 43824816, 43284464, 21022445, 43735367, 41334190, 41159945, 43715825, 43715737, 41089075, 21067385, 41124306, 41083492, 41331703, 2057385, 41055989, 41359302, 21099587, 41372626, 43517911, 41328662, 41336473, 43799564, 40903510, 41335435, 41247118, 41003372, 40978230, 43716693, 21163683, 21068211, 41329912, 43681554, 44190746, 41452199, 40883739, 43201192, 43844067, 40903518, 40847586, 43744458, 41352231, 41332656, 41335381, 40927192, 41413269, 41335379, 41084860, 41191093, 41334699, 41378899, 19125148, 35865048, 41331451, 41335543, 41153777, 40965920, 41021331, 41355522, 43769844, 40717871, 41332382, 35896705, 41229381, 41471999, 43277280, 21153884, 40718164, 40914659, 40146416, 41249178, 2042100, 43209647, 36894758, 40749474, 41277941, 35772115, 40949300, 41099205, 41180279, 41315390, 40228392, 41083479, 920712, 40999369, 43717875, 41315386, 41337118, 41034513, 41354206, 41116099, 41340841, 43186643, 40841336, 40965882, 43625865, 35149112, 41356962, 43201183, 21128931, 41445732, 43644943, 40836292, 41122196, 43518088, 41359799, 40717869, 41061520, 41216061, 41166056, 35745112, 41344320, 36780477, 21070254, 41331912, 21119129, 41280000, 21168411, 40234050, 41210057, 41343819, 41333357, 43861067, 41345314, 41135694, 41332083, 41182937, 44045609, 41258013, 41159941, 44102683, 41120306, 41350772, 43146022, 35201066, 41333358, 43271285, 41034486, 41327710, 40049691, 40041493, 41237349, 40073199, 41005448, 43211998, 41146297, 21094864, 40731080, 41028253, 43808952, 43818169, 41335268, 41247142, 21070258, 40847589, 40729779, 41335747, 41211692, 41136914, 41333068, 41333499, 40233960, 40997179, 41268127, 41346420, 43601719, 35830349, 41184778, 2057383, 43733331, 2057266, 44050993, 43662100, 40843481, 43690613, 43818432, 41128501, 40716463, 41063241, 40917956, 2057376, 40934734, 43177438, 41355463, 40878689, 41395312, 21067781, 41354772, 41153704, 41222207, 43765239, 21168371, 41332287, 36504363, 41334429, 41353507, 21153885, 41343821, 41333353, 40934744, 41145608, 41034506, 41186919, 44190600, 41407204, 40085198, 41439616, 41302992, 41328407, 43595123, 40862770, 41331326, 43582412, 41259300, 40903494, 35780053, 41034514, 41340073, 41401618, 41427301, 43798927, 41351522, 41329325, 21022447, 41333530, 40749546, 21055572, 43601715, 43607783, 40934717, 41137103, 21150014, 41444845, 41335434, 41328951, 43584786, 41279996, 41334695, 35606548, 1551171, 40899010, 41329526, 35148119, 41329497, 40028569, 41330640, 40997222, 43212010, 41337096, 40965897, 43644112, 44076771, 42876078, 21173454, 40947253, 43763749, 21040643, 41336738, 35774491, 36809113, 43708473, 43201487, 41398836, 35767219, 43751751, 41005442, 40941074, 40991080, 41048603, 41126064, 40966268, 35852515, 19098198, 41008358, 35828578, 41386901, 19076136, 43798401, 42479182, 43042597, 41376082, 41247105, 43787736, 41091054, 41186925, 40867950, 41331314, 21138809, 41329705, 43793314, 21130236, 43763332, 41381299, 41149181, 41332407, 43135167, 41277926, 36813805, 41333130, 43752699, 43728187, 19129512, 43799775, 40947265, 42903212, 41345383, 41134794, 43143697, 43680136, 41336918, 41309182, 43843437, 40233961, 35775922, 36895050, 41186924, 40834508, 21048437, 40872545, 43834954, 41026349, 41091089, 40936896, 43860222, 43636595, 40903551, 41328193, 43636599, 46275941, 43255982, 41113061, 43745640, 44198237, 41330393, 44190832, 21065407, 43770741, 2055931, 43600478, 42934821, 41069526, 41335746, 40991084, 41341371, 41363300, 41099181, 43778093, 41332664, 43601604, 43637543, 41249177, 41353770, 43031880, 40934730, 40896825, 41330647, 41216068, 40717872, 43852935, 41344048, 44122028, 40965915, 40717165, 41239683, 41338993, 43668047, 43266787, 43607912, 41057543, 41253164, 41327713, 43218381, 40738862, 43517989, 43680058, 35897665, 40717868, 2034613, 43028388, 43834453, 41331864, 43746701, 40927823, 35851470, 43582411, 40965859, 21138816, 41214273, 40853711, 41247117, 41336471, 43806651, 43174731, 40843473, 43272792, 19129513, 41091074, 36063389, 41335401, 43179239, 44076142, 41333787, 41333776, 36057000, 21104625, 41034504, 43168180, 40836813, 43155453, 920713, 41480774, 41315408, 43799310, 2057378, 40880797, 41412385, 40990366, 41338005, 42939646, 21110799, 40968029, 36896526, 41331963, 43681307, 40717027, 42902111, 41277915, 41439406, 41445798, 40884950, 41124296, 41184734, 41338784, 40976985, 41335049, 19130170, 35745517, 40987457, 41279976, 35606544, 41336419, 41336372, 41340141, 41084142, 40085515, 41337106, 41153762, 41065629, 42934726, 21126743, 19006940, 41212686, 35886731, 41134800, 40839475, 41333081, 41177220, 44192083, 40896209, 41091091, 41237747, 35778133, 19061549, 21099575, 41335566, 41335271, 41409725, 41086457, 2920793, 43746117, 41182945, 40976979, 41224160, 40927812, 40928506, 41336373, 41329348, 41323802, 792424, 21148656, 1518745, 35780728, 41238553, 36895139, 35753752, 41224122, 44131574, 35765919, 41327512, 40962615, 40903529, 19028933, 40915868, 41343265, 41224162, 35779294, 41277916, 41359282, 43183981, 21145863, 35779201, 43727718, 41091072, 41346542, 40847598, 41135699, 41384989, 41331702, 21138825, 43040366, 43626098, 41353342, 41335272, 35886835, 41277950, 40234824, 41330390, 21119134, 21128909, 41352451, 41030376, 41163656, 35777386, 41122220, 41331511, 41335406, 36882258, 41327918, 41333835, 40899017, 43799620, 41359885, 41334025, 40738355, 40987157, 41097168, 41327667, 40887273, 41414898, 35786939, 41369722, 41329055, 41338697, 41026364, 41120298, 40992131, 21065352, 21065309, 42948563, 43201492, 36063776, 40843465, 43816776, 40928400, 40883729, 41146914, 41239684, 41328032, 40872490, 41065609, 43787724, 43201179, 41059500, 35776968, 35778430, 41332329, 41181298, 40841310, 43582466, 21127920, 40110591, 41333013, 41133482, 43726406, 1592249, 41003385, 41146311, 41315387, 36055777, 21085078, 41009590, 43774417, 41342544, 41329914, 41222209, 43135156, 41335267, 41358218, 41028248, 40749482, 41084139, 35851471, 40872510, 41333644, 43824814, 43152909, 41273675, 41186401, 41354256, 43672772, 41289135, 41051903, 43733933, 44815778, 41407525, 36881222, 40718368, 21060292, 43835835, 41418825, 43771928, 41338000, 41426891, 44122027, 40927824, 2057364, 40989736, 41089085, 43787782, 2057371, 35885562, 43197555, 41451317, 41277923, 41413091, 41393239, 40946027, 43663060, 1551046, 41329011, 41331870, 41336979, 41333014, 42934835, 43832162, 36814629, 35777298, 40028598, 40936901, 40835198, 41247130, 41338014, 43823901, 41209366, 43218484, 43862893, 40980292, 44192397, 41307256, 35771094, 41133496, 40866955, 43034982, 41034498, 43609907, 42918692, 40749550, 41042733, 41351381, 41395595, 41331026, 41397259, 41333777, 40959102, 41327641, 36421584, 43726125, 41329004, 41145606, 43031868, 43680063, 1971534, 41182955, 43584171, 43188393, 19034647, 19108506, 21166218, 41028277, 43698012, 41324809, 41197427, 40872513, 41243653, 41220780, 43781012, 43691582, 41329749, 41423248, 41028267, 35776800, 21070278, 41357278, 43763334, 43680178, 44158948, 43196394, 43296713, 43256486, 41337748, 41401705, 44196357, 41146925, 41278270, 43190132, 42934743, 41331320, 41333021, 41209369, 41245197, 43751877, 41335748, 41338696, 40976990, 41311325, 41334711, 43603415, 41408206, 40901661, 21173387, 41367140, 35137634, 43258668, 41328008, 43739518, 40841338, 43806653, 36887877, 41329496, 43745392, 41333360, 41328584, 43843439, 41024729, 43526087, 40986444, 40905135, 21168419, 41099183, 43655962, 41122219, 35897668, 21168417, 41358459, 41357748, 41454554, 41402802, 41452204, 41214258, 40018861, 41417441, 40884952, 43762332, 41392918, 41081278, 41196154, 40972010, 41153717, 41352160, 41339892, 40934746, 41122181, 41197416, 36780903, 40927815, 41342513, 43726417, 41338783, 21138812, 40965900, 40988576, 21089695, 2042092, 41331584, 41253146, 43583719, 40936898, 40849564, 43717969, 40832215, 46276005, 43654655, 43817674, 41070696, 21146755, 40943053, 41370488, 42922443, 43770743, 2042101, 43589896, 35777407, 21128921, 40934738, 41284139, 21107263, 44102605, 43035008, 41134801, 41330198, 41228479, 44190831, 35781108, 2055914, 44097242, 41184787, 41338782, 40718367, 43691568, 40729816, 42934829, 41036475, 44101193, 1366731, 41304571, 41020691, 41339320, 43035012, 41248733, 41354930, 43770739, 35897727, 41304607, 41277954, 36895170, 42942765, 40749968, 41407437, 41399035, 40749559, 41021334, 35606536, 43146327, 43028401, 19033380, 43625770, 40729773, 41135696, 21168416, 41367021, 41247114, 43788574, 35606547, 35776266, 35134764, 40718374, 43709443, 43039532, 43799311, 41331697, 43769763, 41335329, 41397866, 40234054, 2904165, 41030381, 44062626, 40999360, 40870601, 40941091, 43762331, 35830461, 1146889, 790239, 41354725, 41334421, 43841982, 41331583, 43841898, 40073812, 43860392, 41346364, 40060730, 43691577, 40870603, 41020690, 43769773, 41334024, 40964024, 21077276, 41247080, 40725285, 41228470, 35742712, 21176262, 43611976, 43751876, 43769866, 43754300, 40997213, 40897108, 21099574, 41335044, 41340669, 35776911, 42922388, 40738327, 2030884, 43811584, 43690959, 42970584, 40964029, 43147774, 41372958, 41331035, 40853714, 41351757, 43690961, 40903525, 44071443, 41328683, 44088996, 35749310, 43644360, 43862241, 43517965, 41089082, 43853708, 41330348, 41115357, 40978231, 43134958, 41333779, 43716691, 19131603, 43034981, 40715935, 41409935, 40988575, 41336487, 43823810, 44082227, 42934705, 2042097, 43034991, 36278354, 44058447, 44126653, 40990353, 41334157, 41355634, 41357179, 41065635, 41273497, 40909695, 44130116, 35763124, 44190744, 41116097, 43034970, 41122209, 41040664, 41371725, 43024883, 41357577, 44111485, 44190471, 43842027, 41339868, 43135145, 44083095, 41386055, 41030378, 41330396, 41333385, 41309142, 41083488, 40725286, 41146897, 2034626, 43168178, 2054953, 41333772, 35145488, 41329706, 41182929, 41329755, 41320470, 44050036, 21099589, 43212286, 41351589, 41355633, 40073194, 43852582, 43823814, 41363845, 40932768, 41331707, 43619919, 43734865, 41174954, 41354427, 43648159, 41417920, 21070256, 40852474, 40866341, 41247085, 789936, 41309181, 43817863, 36812422, 41289148, 21050468, 40927819, 41394281, 41065606, 36278424, 43190115, 43644116, 41023698, 41336472, 43770738, 40959122, 43269586, 21094858, 41236558, 41339271, 41327705, 41327744, 40729775, 36073991, 40956194, 41330905, 44073107, 41008345, 41339050, 41338271, 43680341, 42948559, 43601888, 41341656, 40865649, 35758282, 41380387, 41008357, 40968026, 36056983, 21163690, 41335965, 43842922, 40143531, 41227239, 43168171, 43157319, 40915851, 40941079, 41177862, 43626777, 41302990, 19075849, 40831541, 44073108, 21126127, 36063378, 41153743, 40893948, 43168158, 43589931, 41327514, 35141817, 40880789, 43218443, 43034997, 41333728, 42629020, 40997239, 43157314, 43625868, 41321709, 42480714, 40718366, 40988577, 19096439, 43157310, 43853864, 41180278, 41337707, 43618170, 40716045, 41335541, 41332885, 41330823, 41345573, 43717228, 35757669, 41329700, 41122205, 41255168, 41320464, 43781407, 41315409, 41332401, 2034705, 41346228, 43039433, 40852470, 44130990, 41150238, 41334430, 41207179, 41247119, 21099570, 41304566, 41333496, 41341780, 40995316, 40878706, 43147775, 41356602, 43769770, 41328815, 41309161, 43275671, 40946033, 40903491, 41328668, 41028258, 41117718, 41366252, 43680138, 41332167, 43028395, 43825414, 41036478, 41334754, 41059481, 19121383, 44165595, 36279365, 21030814, 43166402, 41335088, 40995320, 40843476, 41337266, 42731508, 40941096, 41245203, 41130578, 41184781, 21067268, 2042116, 41059493, 41344812, 41330630, 43638939, 43601429, 43799315, 41253139, 40991081, 41054907, 43618472, 40947254, 1973166, 43841980, 43135453, 44112410, 43805709, 41331039, 19106202, 41117720, 41331968, 41332080, 41197413, 41239694, 43662083, 41413160, 43854280, 41105496, 41120305, 43600466, 43199421, 35777015, 41386903, 2042090, 41327761, 41338534, 40841290, 2057322, 41271184, 41178642, 41402576, 21175217, 35772112, 36074601, 40946035, 35777767, 1510781, 41271859, 43698023, 40997181, 41330628, 41122974, 40995313, 41332162, 41331325, 43798085, 43526416, 41337548, 41336722, 40749555, 35770223, 43751849, 41336078, 41245194, 43031876, 43602814, 41354209, 41249197, 41084120, 43515273, 41196160, 43734866, 43138471, 43662098, 21117063, 41356553, 41330816, 41133480, 40999364, 41335781, 41193123, 41333076, 41153726, 43736036, 43179537, 41178643, 35160730, 41128513, 41345782, 41339505, 44051671, 41105491, 41197429, 44074758, 40872526, 41412424, 35742714, 41034494, 36508432, 19006963, 41335383, 41339164, 40155457, 43212000, 43582778, 41401029, 43674117, 21088477, 21089697, 42934754, 41300323, 36054457, 43185582, 43262691, 41339808, 43823917, 41059482, 21128906, 2057295, 40979102, 41330351, 41277961, 41095740, 36881885, 1511403, 43861065, 41335108, 41334029, 40725284, 43028396, 35897725, 41335788, 41054918, 41329053, 41339169, 21128905, 40718375, 41331983, 40934711, 40234059, 41102141, 40917998, 21106525, 41273473, 43816406, 40997250, 41330987, 41116100, 41099182, 36780569, 41089072, 43716689, 40018160, 40073460, 41309170, 41026351, 2034610, 41247143, 41331695, 41292462, 41329499, 41333671, 21055583, 35777563, 41330401, 40965856, 40943084, 41337088, 43853850, 41178635, 43288533, 21071597, 40880801, 41375036, 40962475, 41334465, 21148657, 43168168, 2031891, 41346114, 41335587, 41337937, 40234039, 43726774, 2034615, 41020688, 41168095, 43805978, 36035926, 40836271, 43024863, 21050456, 40972022, 35786568, 40946043, 43751837, 43028387, 41337754, 41332397, 41344504, 41333086, 19130164, 41084150, 40822869, 41332549, 41353210, 41228466, 21173458, 41355393, 41269501, 41346298, 40718928, 40831880, 2057245, 41030380, 21106521, 41334867, 44030688, 41122214, 41345781, 41166064, 41335330, 41334981, 2055935, 42482500, 43799566, 41336075, 35151928, 21039871, 21070255, 40997232, 43157577, 41103361, 40915867, 40947259, 41004267, 41343531, 41336418, 41329530, 41336464, 41332330, 41358723, 43637542, 43031872, 42482606, 40997201, 41365904, 40980299, 35147172, 19065985, 41020694, 43698065, 41352927, 41336300, 44033232, 43697928, 41153710, 36884767, 41355464, 41124300, 41336622, 40725282, 41337107, 43682128, 40822394, 41337352, 41354823, 40738344, 41054321, 41389105, 41387037, 43662172, 43279544, 41236314, 21153881, 41034492, 36054466, 40107134, 41344651, 41333345, 40918936, 42483158, 42942778, 41184729, 40716469, 40841831, 43680180, 41001935, 41343468, 40928507, 43717184, 36272971, 41333727, 41340580, 43690448, 41196147, 21168403, 41340641, 41151884, 2057373, 43628649, 41412411, 43629880, 35776528, 2030655, 41337758, 41337521, 41344737, 2031880, 43744475, 43698045, 19006965, 21100967, 43277281, 40738835, 2935780, 21138828, 41328773, 41261309, 40941082, 43860394, 40934748, 41330626, 41338785, 21148673, 35830465, 41277952, 2012255, 43727429, 40932762, 40855737, 36884998, 43145833, 40823662, 41022030, 41328001, 41335779, 43765240, 40965895, 21130234, 41081183, 43860592, 40965877, 43212004, 21021100, 41097171, 36785024, 43656083, 21035963, 41122218, 40936911, 41216074, 41284159, 40749473, 40244451, 43780409, 40027454, 41178637, 35758482, 40997245, 43698044, 40976992, 41331206, 44172405, 43034975, 21057550, 40972020, 41401935, 40749463, 41102124, 21163562, 43157288, 2030657, 41184786, 36260358, 21070270, 44034851, 35886834, 21060304, 40855734, 41337358, 2920797, 41336532, 43152419, 35762674, 43644961, 43214349, 40234043, 1511404, 40997589, 41338008, 43718198, 41344654, 41124298, 43698949, 41481441, 21045800, 41328678, 41332336, 41120311, 44089670, 41228472, 41269243, 21061672, 41240349, 43273660, 43836345, 43783373, 41082865, 40732457, 41332131, 41181306, 41208721, 41057534, 41332662, 41352164, 42934862, 41359987, 21079969, 42480350, 41070717, 41174314, 41240932, 41387168, 43654643, 41268431, 41332684, 43805814, 41329627, 40085506, 41239686, 43664387, 43737568, 40843457, 21158621, 43177446, 35886837, 42723204, 41328013, 35766791, 40732464, 41290399, 40934704, 43739423, 41358301, 43212009, 43841900, 43636600, 21158618, 702629, 21070248, 40880974, 40834510, 41357873, 41335402, 21165991, 40725707, 21067389, 41358191, 40915863, 41333451, 40901651, 41167354, 789930, 42942787, 43751848, 41481608, 43146044, 40235167, 43220661, 36262332, 41344227, 41298715, 43739507, 43190133, 40717866, 21109440, 41228465, 41230585, 43190426, 42948535, 21099580, 36275714, 41115990, 40112724, 41115367, 21166201, 21099578, 42712579, 40901660, 41402753, 41059491, 41344653, 40824813, 36810564, 41008334, 21118342, 40990347, 41359884, 43686656, 41222177, 41330388, 21089670, 44176843, 41276172, 41320488, 40234038, 40976999, 43693433, 41028273, 41342861, 41333739, 43733953, 43031879, 41358195, 41346726, 41338015, 40874698, 40961460, 41341963, 44112407, 42614329, 41329758, 43603723, 40978247, 43662170, 40956196, 43163940, 40965890, 41343904, 41181314, 2057314, 41337702, 41333018, 41341589, 19106693, 2031943, 41059449, 21085072, 35786641, 35742422, 44033994, 41340366, 41331089, 41039426, 41270541, 41439275, 41337487, 41336205, 1719021, 41091062, 40853717, 42934834, 41370489, 35132184, 1972872, 2057239, 40231769, 43201484, 40824220, 41376090, 590125, 41336866, 1551007, 41182940, 41356513, 21134234, 41356894, 43276827, 43775316, 40915855, 43680197, 42934848, 41284164, 21168407, 19106652, 41334416, 40905637, 41330251, 41407381, 41255145, 43219434, 36037491, 41054910, 41153716, 43847895, 41335097, 41289143, 43260786, 40085186, 35150648, 41070714, 41097181, 43024875, 44063999, 41199512, 2920799, 40865807, 41329318, 40729245, 43516066, 40934701, 2034621, 2057372, 41336051, 35606535, 36889955, 43163560, 41335966, 19106692, 43680342, 41392246, 43589819, 43684587, 41329064, 41271853, 43603418, 40872488, 40903487, 19067727, 41344983, 41065637, 41258023, 41363909, 41122187, 41089080, 41093116, 40234060, 19007009, 41338832, 43726773, 41336740, 41330989, 43141513, 41342548, 41289128, 41336477, 43680062, 43836961, 43817679, 40732466, 43157297, 40946029, 41279985, 41337524, 41328409, 41352271, 41301630, 41240325, 21168395, 41249186, 21079970, 41166066, 40824398, 40914656, 21099571, 41008328, 41030388, 41358909, 41291675, 41114017, 43135457, 43675129, 920917, 36063385, 41317504, 35886735, 41332659, 40872547, 41345934, 42934855, 41228487, 35761829, 19130122, 41290395, 43219431, 41367322, 41276152, 40841324, 41177867, 21150013, 41334197, 43637994, 41345505, 41335568, 44188109, 41353504, 21099596, 41327755, 41338810, 41329062, 41269931, 36055499, 41276154, 41054320, 2030658, 21138815, 43212007, 43630488, 44089589, 41337753, 41357671, 41284147, 41329112, 41352675, 41292457, 43687761, 41412951, 43618515, 40716048, 41219861, 41351323, 40729263, 41331696, 41327754, 41337708, 40749448, 41021601, 41039421, 43219121, 41009582, 41330746, 41115726, 41184755, 43783374, 41337519, 35778361, 41336426, 41327666, 41355467, 41179293, 41346299, 41276160, 41336672, 41302154, 21068210, 2034622, 21089680, 41086447, 21070260, 40972015, 2057320, 40979103, 41122180, 40934724, 43690605, 40941070, 41330430, 43752702, 41071926, 2030885, 41333737, 35776265, 41059983, 41327511, 40915861, 41381902, 35750568, 41354408, 43144270, 41059984, 40234044, 41284158, 44031345, 42934713, 19027187, 42934708, 41339786, 41328578, 43853863, 40947248, 42934735, 35777012, 41255154, 41356464, 41335089, 43609250, 41338047, 40934716, 36057014, 36056984, 41353315, 43774940, 35409725, 41099186, 40934702, 41361084, 43708826, 35780495, 41159958, 41166055, 41222188, 21026103, 41338368, 21173469, 41341657, 35753476, 40992701, 35606532, 36405399, 43585359, 43607876, 41059467, 41332400, 36781614, 41352369, 41022671, 41126081, 36055497, 36063799, 41277932, 41331966, 40729800, 41335103, 41258026, 1551100, 41153776, 35141583, 44176282, 41166058, 41177872, 41333946, 40841300, 42934724, 35130803, 40965875, 43162091, 41329351, 41334226, 41354409, 41331047, 41372336, 21060277, 40073809, 21051870, 42873418, 36418762, 41335087, 41341987, 41337185, 41309150, 36895363, 41367419, 21109438, 41116731, 41327639, 43781655, 40872540, 40932767, 40843470, 40993738, 41337743, 40899011, 41216047, 43663550, 41247098, 41473539, 41245208, 21070275, 40909694, 41273499, 43841896, 40853724, 41351840, 21061661, 35143600, 41336636, 41072807, 43526420, 42941639, 2031890, 35776146, 2034637, 41336074, 35828579, 41086448, 35830393, 40738326, 45774916, 41301635, 41184790, 36269533, 35758030, 21128879, 40716043, 2055906, 40964013, 41332342, 41122215, 41333729, 35765917, 41365033, 40028604, 41003394, 21037916, 40725273, 41253161, 43734140, 41300055, 43708658, 41184783, 41153728, 36053951, 35745117, 41230580, 2030889, 41279521, 41334030, 43201181, 41334386, 35780600, 42715234, 43024872, 43715819, 35157875, 21169761, 43854282, 41153742, 43662167, 19107677, 43656429, 40823642, 43207826, 42921164, 40085180, 19126169, 40749484, 21166401, 41315405, 43780794, 40909700, 40835879, 41153714, 41146307, 40903521, 41328676, 42725934, 41117726, 43185869, 41329954, 21116758, 43656088, 36508822, 40234797, 44124954, 43196552, 41350826, 40738329, 41340513, 42707634, 41332263, 40993737, 36037492, 43805770, 21148665, 19106829, 41209940, 43818167, 41353473, 41028237, 44076705, 41329764, 40884953, 41323807, 43861394, 41445844, 21079975, 43631584, 36278661, 40749485, 41277945, 41344501, 44073787, 41058984, 44135657, 41338206, 41034497, 43680972, 40227495, 43644115, 43709042, 40729770, 40979500, 41247136, 41330841, 43727781, 41115351, 43207402, 41008337, 36056996, 1592259, 43698017, 41103351, 43133400, 40928509, 41339321, 41218186, 41353209, 21099519, 41388459, 35780091, 41390991, 41222187, 41407205, 41049714, 41342948, 42934707, 36509786, 41102125, 41289142, 44197630, 40917953, 40898089, 36891618, 41338120, 41332867, 36257150, 41330006, 41338006, 41184780, 41334710, 41126082, 41331540, 41133485, 41052527, 41341654, 41290392, 41358721, 41337712, 19129289, 41337371, 1551122, 2057242, 41356514, 21168412, 41279994, 21097581, 41290400, 40233948, 41255167, 40865637, 41329218, 41284166, 41146299, 41329268, 40834518, 41135701, 41024721, 41415247, 41410081, 41329921, 43697930, 40955071, 41337547, 43630096, 41003389, 41039418, 43601437, 21061674, 41184743, 41352697, 36895250, 41034501, 21128918, 43034986, 41410117, 40870606, 41216075, 44164823, 41330193, 41342325, 42934747, 19007006, 41331989, 41022672, 35754792, 41334907, 44198091, 35885742, 41330003, 21071598, 21099603, 41342512, 41253160, 36265338, 40823683, 40717022, 40915865, 43271752, 43644327, 41341345, 36278362, 43842876, 41128521, 43277282, 41216082, 41331453, 41311336, 43157298, 35148983, 19067795, 42718618, 41337268, 41356162, 2055005, 41161943, 41328005, 41091088, 41407799, 21021104, 41209380, 43631935, 41337101, 21155810, 36259287, 21119141, 41184736, 41341115, 40903511, 41061506, 41334023, 43805691, 21175334, 42934733, 42479179, 21021113, 41153709, 41253147, 40991751, 40749434, 41400209, 41114706, 41303643, 41227221, 43024868, 41105494, 44064010, 44034849, 40847602, 40909686, 43269633, 41358066, 2057382, 41247102, 40997198, 36888030, 43738022, 36896546, 43655963, 41346727, 41277949, 41329741, 43135168, 41065605, 36056993, 36277910, 36063780, 41134788, 36419765, 41332817, 40738863, 41365628, 21057418, 35784836, 40877247, 41122208, 21140166, 43823824, 21021099, 40738325, 21097606, 43860169, 43201488, 43744462, 43024871, 41472066, 44176210, 43864775, 35142894, 44099637, 41408453, 43751878, 40738359, 40835199, 40911821, 43791319, 43727715, 41329696, 43654630, 21158598, 35777051, 43805983, 40914654, 19132464, 43715816, 41247113, 41336421, 40725275, 40749535, 43770295, 41333568, 41155873, 21050466, 41330385, 43141486, 43691874, 36419768, 41337141, 41410116, 40958444, 40930137, 21061683, 21167583, 41329220, 41089076, 35604736, 44084340, 40914643, 41020698, 40903489, 41281732, 2920795, 40961458, 40234095, 21138777, 43680971, 43802903, 43157301, 41208726, 43590805, 40234805, 40738846, 40725269, 36814213, 41117116, 43202828, 43601438, 40972023, 40943082, 43627271, 42970564, 41309158, 43028392, 41335102, 35144017, 35606533, 41276166, 21079956, 43190432, 40934720, 40948531, 41097162, 36885754, 40878691, 40749439, 44058614, 41029889, 40997590, 41315410, 41052913, 21165502, 40718379, 21065405, 41336422, 42953129, 41032075, 43737575, 41333775, 41353505, 41329913, 35604729, 40725274, 41182947, 41407600, 41161944, 41083490, 41334022, 40914664, 21040709, 40959116, 35161586, 44050994, 41330191, 35897678, 43844034, 21128915, 43788577, 41143995, 42970566, 41472711, 1551006, 21128903, 35152740, 41097170, 40883736, 41018142, 1831858, 41164811, 41334871, 21089621, 21091044, 43626772, 41289133, 40903856, 35830394, 36063373, 43698097, 40997242, 41330001, 43860175, 41028255, 41331030, 40729246, 41330007, 41218176, 41084149, 41334243, 41379481, 40878695, 21050455, 40935091, 40941088, 21065406, 41358912, 36063380, 44100657, 21094850, 41216062, 41338184, 40905137, 41028247, 40112721, 41054917, 36266078, 1719012, 40749970, 41128496, 41245198, 41003393, 40917951, 43698957, 41354606, 41335409, 44034263, 41329223, 41327389, 41277935, 19130167, 41331670, 41328682, 21077647, 41311326, 43853795, 40874709, 43805774, 41331667, 41021312, 21058369, 41153732, 43843438, 40934741, 41332340, 36266434, 41356421, 41054914, 40896828, 35153456, 21120475, 40725265, 41021319, 43690602, 40898090, 44180041, 40903522, 43781863, 42941666, 43028400, 2034618, 21050465, 43835185, 40900079, 41354306, 40959120, 43594451, 40865645, 40934733, 42482155, 41036461, 41345784, 43645426, 40915848, 41329752, 41360273, 43146023, 41352367, 35776185, 19050908, 21167581, 41336818, 36786125, 43734857, 41059463, 41329742, 40991744, 35770229, 41335499, 43034976, 43681463, 40878683, 35787493, 41364633, 40958452, 40903532, 43852578, 41333070, 41332868, 41337083, 41402191, 41217727, 41352808, 43518139, 41153722, 36269901, 40932765, 41340265, 41333500, 41180288, 41197423, 41184792, 43656420, 41052519, 41053296, 43141524, 41212660, 41250844, 42948552, 41289140, 43157285, 41149185, 41357832, 41358821, 41320471, 43135155, 41359382, 41342859, 40841327, 41321704, 21050470, 41097156, 41117721, 1592245, 35604857, 41333778, 41084132, 40909688, 41335101, 40225703, 42970587, 40841342, 2034704, 41222198, 35130847, 21077646, 41180291, 41336077, 41143994, 40949298, 41057530, 41247454, 35781045, 44069636, 35866489, 41331865, 40729268, 40738859, 40941067, 41153759, 21089692, 41303660, 41332324, 40073437, 41164814, 35763123, 21021115, 41327492, 35157817, 36063791, 41159954, 40941066, 41332335, 41084857, 43276828, 40999361, 41328150, 41210051, 41040666, 21081310, 19107561, 40717865, 41358064, 41402728, 41354431, 41345572, 40990956, 36277952, 43197689, 41134799, 40980294, 40965917, 41070699, 41093125, 35757658, 40870598, 43034969, 41212671, 21140169, 41337488, 43034980, 43644020, 41164801, 40839473, 789934, 40965898, 41355978, 35754795, 40979100, 41184754, 41341986, 21126128, 43583436, 41341690, 36421583, 41336057, 43600467, 2057362, 43823807, 40738856, 40839467, 44198588, 40896826, 43734867, 41410015, 43174705, 41247144, 40997197, 41153708, 43662175, 21070277, 41357069, 40915857, 36056994, 41267259, 41416949, 40995294, 21165503, 19040242, 42934706, 21070257, 41351216, 40917955, 41327772, 2910253, 41189609, 40878680, 43636926, 41304756, 35776143, 41241038, 41334048, 41337257, 41184785, 44072824, 43517864, 40732458, 41360177, 41279987, 41331960, 42934854, 41327753, 41028242, 35897674, 41444840, 40823887, 43662101, 40861708, 40234053, 21145866, 43744502, 41379723, 35779607, 41337799, 41333806, 43680969, 41330865, 40824087, 44108283, 41327541, 41332008, 21097156, 2042099, 43716044, 40847574, 42934837, 43665325, 41331979, 35753897, 40874694, 41451331, 41216087, 41196156, 35776221, 44192307, 40929192, 35897682, 42941640, 43680343, 36277811, 43798374, 43798412, 41276171, 41331753, 41331315, 21140173, 43602199, 43727713, 41258016, 41337307, 43800681, 41483048, 44189950, 43824822, 44057663, 41335053, 43168478, 41331694, 41334388, 43594450, 41303655, 40085503, 19034014, 40959109, 21128902, 40991077, 41331585, 2054954, 41342396, 41345675, 40976982, 44167035, 43757433, 41059476, 41355520, 21050487, 41003400, 41332444, 21045746, 35778350, 21150019, 40915870, 41259298, 41102118, 40073807, 41336474, 43727426, 41147611, 41117722, 43823826, 43157327, 40149642, 21061671, 42629018, 43644019, 42727277, 41413252, 41332560, 41328194, 41330014, 43824109, 43745807, 41028243, 40227406, 41335051, 41343071, 43212003, 41255157, 40872541, 40927818, 41331663, 43655633, 2031939, 21061655, 41335411, 21168408, 40978246, 43632310, 41331709, 41099199, 40841294, 43636927, 35776815, 43771164, 41334751, 43141525, 43042474, 41350622, 41336743, 43734871, 41332815, 41327665, 41276153, 41355392, 41112155, 41184740, 40990368, 40887057, 35897663, 43709438, 41237129, 44083850, 41331292, 40847579, 35787365, 43655900, 41330983, 36781245, 2039854, 19132094, 41344600, 43608767, 41408412, 42970598, 40234096, 41343992, 41333726, 41011635, 40887055, 41097174, 40972028, 43608763, 40884947, 43824810, 43656424, 41255155, 43142159, 41346546, 40914663, 41351753, 41357974, 43805690, 42934859, 42479771, 40870593, 2034609, 41091049, 41483488, 41331196, 41364178, 43842031, 41028266, 35745516, 41334890, 43201190, 42948564, 40903507, 40897486, 41284141, 41199684, 41346112, 43600172, 41115371, 40934755, 41345506, 41329911, 19107562, 41196158, 43654651, 40234795, 21140175, 41334833, 41345655, 40749481, 41277927, 41311337, 41354721, 41086450, 43829335, 41039424, 43600461, 41427004, 40997238, 41334387, 21058368, 43164313, 43146047, 43719880, 21028110, 41330818, 43663052, 41330652, 41327466, 41315389, 44112411, 35776186, 43745397, 43592360, 43593025, 41337742, 41042725, 41093131, 41408517, 43781033, 43146017, 43656671, 41352270, 40965916, 43823599, 41061517, 35745782, 43039461, 35139303, 41329494, 41018426, 41011634, 41331704, 41366512, 40865014, 43834625, 41334605, 41292463, 40895056, 43139985, 41331705, 41128520, 40997220, 40841299, 40233952, 41343045, 41332818, 40864373, 40883724, 43190427, 43160150, 41228469, 43153274, 41335042, 41334020, 21107539, 41229775, 43179233, 41330020, 43690960, 40915864, 41343720, 41222180, 19067730, 43601930, 40865806, 40825241, 41329525, 40883743, 44101649, 21050439, 43709779, 43857150, 41361325, 40839457, 35757668, 43806657, 41302645, 41353900, 40932760, 43826330, 36781607, 36277967, 2039819, 41330004, 41354328, 21068306, 41222191, 43788583, 40841322, 40988567, 41084145, 43177443, 41362900, 41227232, 40843469, 36507430, 41352366, 41339894, 21021102, 43626985, 40866229, 40839449, 35886739, 40922854, 41307264, 40961461, 41330982, 35749311, 41338046, 41182936, 40738860, 41356694, 43841895, 41457910, 2057381, 41084733, 43135159, 41336816, 41122211, 36055505, 41342516, 21070273, 2055913, 41330906, 44086626, 41166057, 41356310, 44060312, 21156728, 41338834, 41133475, 41388859, 41039423, 41327760, 43526419, 40979499, 40909682, 36886654, 41330394, 41389245, 41372164, 42934858, 40725268, 41356079, 41328684, 35827996, 36890864, 41209657, 41320483, 41337900, 41333363, 41456156, 21130240, 41328123, 40865639, 21140183, 21055573, 21038287, 41309185, 41330549, 41371703, 40073811, 41336867, 41150242, 21104723, 44197663, 21040711, 40839456, 41030398, 21156450, 41315404, 41093139, 41311342, 41224163, 44050991, 36812505, 40717161, 41335789, 41328712, 41333947, 41356693, 41197425, 43787505, 41018416, 36811923, 41345845, 35754791, 40738342, 40839469, 42942766, 41216098, 43629639, 43630487, 36503395, 40841301, 40997190, 41184741, 41159930, 40738845, 42934805, 43806650, 43600453, 44131069, 41212673, 19047963, 41357637, 41355635, 40049700, 1592261, 41346178, 43841987, 41336481, 41327746, 41412767, 41327790, 41022679, 43733941, 44197904, 43157313, 40965924, 43752692, 41163657, 41337305, 41005431, 35150344, 43039530, 40947258, 41018415, 36275565, 41336237, 43680246, 41407380, 43655646, 43674107, 43733938, 41311335, 41354045, 41065610, 41240338, 35754342, 41240508, 40911817, 41227231, 41227235, 40841833, 43157462, 40915847, 41147485, 44070225, 975505, 40841311, 40934723, 43157293, 42939642, 41361655, 41346539, 40895147, 2042093, 21061657, 41277960, 35784815, 41330011, 1507706, 41412641, 41196159, 21045801, 43141526, 41216070, 41103367, 41093135, 40968034, 43685599, 41226022, 35746407, 21091039, 40978237, 41166959, 40934722, 41261308, 43817755, 21048634, 41008352, 43682755, 43861276, 43601718, 43715760, 43625765, 41329106, 40717164, 40824911, 41339390, 2042087, 43698096, 41359990, 35758283, 40964028, 43269686, 41184772, 40896808, 41028238, 41329326, 41128522, 41359800, 43042472, 41007171, 40943276, 21079960, 43166411, 21148651, 43691184, 40060701, 35778711, 41334836, 44030023, 41329061, 43715830, 43716695, 19034231, 40749456, 41332170, 21068030, 41337086, 21128904, 43219876, 40852476, 41222208, 40997214, 40901659, 41222205, 41413190, 41054912, 41329455, 40905640, 41341343, 41028241, 41302305, 21079954, 41178640, 40717803, 43852756, 44109249, 41359130, 41413191, 41412361, 40852478, 41338835, 40978245, 41338992, 41161942, 35886730, 41102128, 41330392, 43647826, 43810366, 40930144, 41114702, 35604732, 43179223, 41328373, 41339272, 43609093, 43715761, 41079962, 21133906, 41066537, 41028288, 44173390, 41084151, 41337982, 43787646, 41353052, 40841334, 19029973, 41329068, 41328374, 41339560, 41008338, 41309200, 19026063, 41336489, 43799619, 41228489, 43157287, 41084856, 19122782, 40234046, 40843468, 2034697, 43201486, 41402677, 43601419, 21130242, 43692013, 41042729, 41065616, 35777176, 41333243, 21098772, 43039463, 42873421, 43741209, 43854274, 904510, 40965904, 41050801, 41093133, 41034770, 43789855, 40849757, 21027991, 2935782, 44064000, 41452194, 41408984, 36063793, 41334418, 35778502, 41372163, 35774492, 41352805, 43618518, 41392025, 41334415, 739917, 43211981, 41333244, 40073533, 43032966, 42934870, 41216054, 41359823, 42948530, 40085182, 44060938, 40173366, 21169765, 35146170, 2034611, 40841339, 35753479, 2057253, 41290389, 41337046, 40972011, 41330544, 44196382, 42731235, 41331982, 42876494, 40965914, 40934742, 40749488, 35753473, 41284127, 920947, 41216057, 41211695, 41335743, 43805982, 41344740, 40972040, 41034502, 41164804, 43157300, 21030822, 41335407, 40941055, 40997189, 42922375, 41333740, 43680059, 41177865, 40855735, 41333786, 40884941, 21061673, 40903488, 41329012, 41328375, 21126129, 43680040, 44199310, 42934863, 43674559, 41338698, 43823902, 41407179, 41299422, 40028574, 43152502, 44159802, 41116738, 41402823, 41216081, 41359586, 21070244, 36063371, 43726122, 41352302, 41333449, 41124304, 41302312, 43835207, 41093121, 41336375, 21030828, 19105911, 40167748, 41472544, 41329916, 40874701, 41276174, 43826081, 40965906, 43146043, 36781218, 43189952, 40931209, 41338176, 43620191, 41277918, 40233172, 43843394, 41337704, 2034695, 2057285, 2034599, 35852514, 44192043, 43291391, 44082649, 43715754, 21140182, 41341171, 43164633, 35776508, 41155869, 43746119, 41413159, 43698073, 41199510, 41334910, 43591386, 35142838, 43167980, 21140180, 40901662, 40738345, 21175332, 35754797, 40073810, 2039822, 43711679, 41261306, 41216110, 41186922, 43650398, 21079965, 35780900, 40028866, 40964034, 19135075, 41355636, 43782167, 43607797, 43801292, 35132710, 40961459, 41212681, 41328677, 40996406, 44192256, 41351326, 41122213, 40872528, 36505595, 43661832, 41323797, 41355066, 41345900, 41327752, 36054458, 41357776, 41091041, 44101192, 41153725, 40896829, 43135171, 41284142, 41361448, 35148358, 41367420, 40861538, 40749483, 1973094, 41277944, 42970581, 41343582, 40717162, 41361446, 40872544, 40725270, 43625928, 43582776, 41302317, 41336742, 41340331, 42941660, 40931335, 21119120, 43662218, 41352368, 41049719, 19067728, 41218180, 42712829, 44128584, 41039436, 2935781, 40852473, 41122717, 2057340, 43654639, 40905625, 40834507, 2904167, 41097606, 35606549, 41336052, 2039848, 41330992, 41412810, 40018870, 41268427, 40903550, 41054920, 41052386, 21138799, 40729271, 41309195, 41178064, 41327427, 43038712, 21061669, 41333131, 35778628, 41330994, 21047742, 2034628, 41327450, 35786936, 43744803, 40749967, 41356942, 21104743, 41329753, 43788572, 40997180, 44125579, 1832264, 19125246, 41086451, 43769839, 35745121, 41097157, 43270119, 41102138, 35406989, 21051863, 41057545, 40732461, 42934825, 41344738, 43035005, 43619775, 40234809, 35746405, 41059450, 44136355, 44130214, 41097172, 43268146, 40914652, 41317495, 40234055, 41385157, 740264, 44191798, 41218183, 43770065, 43787656, 41020687, 43134954, 40968021, 41216067, 40234807, 43135323, 43823852, 41270549, 41330398, 21096788, 41329747, 41358170, 35141498, 40938651, 43709437, 41155872, 35140198, 36056991, 43841988, 41309174, 43727430, 41456400, 41067728, 41277940, 40956348, 35771555, 41357672, 35897676, 40019102, 43585360, 36063794, 43028393, 42948566, 41338002, 41222204, 42934831, 40738867, 41331319, 2919319, 41237414, 41315388, 44184258, 41328686, 41327719, 41160199, 21116741, 41247097, 2034698, 43158931, 2057251, 41116742, 43042596, 44130319, 2012272, 41224138, 41425255, 43646123, 21037913, 40999359, 2057269, 40997209, 41147612, 43602810, 41228482, 40233946, 41214266, 19104630, 21055575, 40849568, 21138779, 43715759, 40946040, 21089686, 40997241, 42873631, 41316323, 36895231, 40911813, 21119137, 40841832, 41339171, 2031942, 41339445, 41033033, 41368504, 41408040, 2034620, 44173954, 41472942, 41343655, 41332928, 2030890, 21071595, 41327461, 40888042, 43654656, 43182931, 40997192, 40028601, 43164312, 40019090, 43168179, 41328774, 41186933, 41342860, 35884665, 1366732, 41410204, 41361601, 21035962, 35606540, 44190277, 41331965, 41218190, 41351637, 41315402, 44044099, 40905634, 21089681, 35780548, 43644018, 41332556, 21168409, 21166009, 40999370, 43024864, 41118753, 41331662, 40997188, 43755946, 40119172, 21119119, 41337143, 41059461, 41161940, 41074063, 43806648, 41332816, 41222266, 41153769, 21140184, 40825576, 41473035, 37003556, 41268430, 41309156, 43860077, 43733848, 40841341, 43751749, 44189951, 41039413, 41407517, 40822226, 41337713, 41327703, 43589905, 36277649, 41003382, 43698018, 41330977, 43609258, 41020682, 21119131, 41354800, 44128016, 35830348, 44182292, 41259304, 21030820, 40725710, 2054959, 41333528, 41118735, 19067752, 41054926, 43649273, 21059466, 789935, 41153734, 41336264, 41042732, 36420025, 43787935, 43847543, 2057313, 41342560, 41342683, 40874707, 40878704, 40852483, 40824816, 43716698, 35829063, 41329904, 41197428, 41343471, 43135160, 35772114, 43212285, 21130233, 40749455, 41059498, 40234097, 41239679, 43663385, 19127775, 41391209, 40872522, 43594452, 42939682, 41152938, 43210209, 41336894, 44191570, 41226023, 41419084, 41338050, 36276620, 43672811, 2030652, 40990352, 21028486, 40872484, 41329319, 41329052, 41365295, 43593632, 42941658, 36417901, 41099180, 43772567, 40909693, 41328174, 41128475, 41334466, 35416660, 41409723, 41053973, 1592182, 36279071, 43168175, 41335109, 41327747, 43697947, 41327919, 41359194, 21050458, 40997227, 2031936, 43726456, 41271444, 40852469, 43834662, 21027998, 41356695, 40884954, 41327910, 41329704, 40234100, 40909683, 41331408, 41360871, 43179243, 41216090, 41332959, 19048691, 43174661, 41334713, 40872535, 43517860, 43762049, 41329762, 43146021, 43727423, 43188399, 41412988, 43726414, 41340947, 35897669, 41181311, 41342323, 21057548, 44190263, 1551193, 40749462, 40909705, 21158604, 43672770, 41222212, 41240332, 41327704, 43207439, 41355902, 43190134, 35154711, 40943063, 21098588, 40995295, 41337757, 35778360, 43609079, 1550717, 41336478, 41224137, 21028113, 41120300, 43636588, 41214261, 41327543, 41354911, 42948515, 40927201, 41239690, 40732465, 21078616, 40841337, 41329007, 41247095, 41331300, 43816403, 41105499, 41328680, 40841315, 41328838, 44031077, 43157321, 21042133, 36054452, 43842885, 43715815, 19019158, 41334715, 44035923, 41284154, 43282560, 40073806, 40901671, 43526139, 44191716, 2031937, 41336000, 43208130, 43200993, 35129814, 40980298, 41333362, 41185081, 35156129, 41354043, 43146020, 41327759, 43135455, 41009587, 41153715, 41307253, 41329009, 43211785, 41376108, 41329229, 36118609, 43263175, 41122188, 36036036, 41122173, 41197419, 41155866, 43645418, 41336469, 43708480, 41309147, 40965887, 21128899, 35776006, 43199417, 44199207, 41245206, 41419042, 43607799, 35886838, 41184737, 41338122, 41258027, 41331197, 41065611, 43824813, 2012259, 41228473, 41329701, 40887050, 41321698, 35886836, 43805766, 41334047, 42712890, 41093137, 40861608, 43655644, 40843474, 43806660, 43818168, 41334868, 43691587, 40878710, 40872519, 21136802, 41358365, 40738341, 40852481, 41304565, 44086621, 41271199, 43583717, 40738322, 43626774, 43146329, 41307255, 44089584, 40932757, 40872848, 41420058, 41331964, 41337902, 41289134, 43782820, 41332405, 40738847, 40841296, 41103364, 35746933, 36780571, 40909685, 43809020, 19016866, 43674558, 21067261, 40018148, 41319367, 41184770, 41269507, 40897109, 40897489, 21128901, 41159949, 35131913, 35865047, 21168400, 43135165, 41247081, 41128502, 40233219, 43201194, 42934710, 43034967, 41356851, 41271196, 40732468, 41271856, 43600471, 41340581, 40905629, 41330380, 43607800, 21068027, 41330389, 41336059, 41335105, 40839455, 42918691, 35140332, 35778646, 43144261, 41057539, 41240347, 41059475, 43736003, 43024876, 40863868, 40966453, 43207308, 35142792, 41053972, 19009119, 19067754, 41186926, 43782170, 1510437, 2057306, 43716048, 43625870, 40738842, 41191103, 43753900, 21040716, 41330015, 35784790, 40997229, 41413286, 21150018, 40154093, 35604738, 21061684, 40914646, 36054454, 43626776, 40824712, 43174553, 43626769, 43788576, 41093124, 35155671, 41159920, 43657602, 41344874, 41005443, 43698013, 40018161, 41336267, 41328920, 43260783, 41359193, 41334377, 36887071, 43625772, 40171334, 44170776, 2057286, 2055918, 41330000, 41008333, 41059982, 41336488, 44816139, 35150213, 41356939, 41311338, 41155863, 43717824, 43834657, 36055504, 41191078, 43741685, 41327789, 41329372, 43163558, 43285940, 35758904, 40936908, 40965896, 41083489, 43609146, 35765905, 35604853, 21138807, 41270534, 41003373, 43864305, 40947263, 43791329, 40965863, 41344229, 35143690, 41356378, 21070243, 41134795, 42716567, 19108479, 40847603, 41034517, 41304573, 21158592, 40959117, 41331455, 41413024, 40852491, 40899022, 40834513, 41336163, 44172569, 40852477, 40965921, 41290396, 40975824, 21130238, 41284132, 43799313, 44172406, 43185409, 40880790, 41247104, 44114493, 35779429, 41351412, 40825281, 40073193, 41331666, 41333355, 43041651, 41339270, 35774490, 1550560, 41335954, 40853725, 41328404, 1972353, 41084863, 21051871, 43041654, 41315403, 1551101, 42713870, 21120474, 41153733, 43715824, 41116102, 41336076, 41212683, 41030397, 41153724, 41315399, 41247106, 41073234, 40896208, 44189014, 21050461, 41122210, 43698025, 43698951, 41376192, 43645766, 43293414, 41299424, 2031893, 41353509, 40732462, 589180, 41093138, 41352178, 42481959, 21148590, 43619883, 43168355, 21040702, 41328796, 41329221, 41136911, 40872515, 35140022, 40028266, 40903499, 41354208, 41327714, 2042098, 36275715, 41328665, 35763122, 41340267, 43861537, 41346037, 41222215, 21143813, 43157317, 35781295, 44064006, 41341655, 41340775, 41240324, 21026164, 40729267, 41331582, 43763326, 40957792, 37593804, 41153764, 21035895, 41222210, 41216071, 41268429, 41335376, 21168393, 43645417, 41366565, 41149177, 41003368, 35202020, 41133494, 41245193, 41359948, 41166060, 44197843, 21158595, 41304537, 21060314, 41103363, 41289129, 43789071, 44101651, 41133495, 41360200, 41341691, 40960511, 41182953, 40934762, 40749459, 40725266, 42934753, 41337081, 43863778, 21050457, 40915862, 41273504, 40935291, 43727422, 41359989, 44195366, 21085266, 2012258, 36895174, 21176069, 35749314, 41335744, 41224146, 41360223, 41309149, 41093117, 41407260, 35145109, 43773984, 44173585, 41340778, 41132326, 19001448, 2055908, 41059497, 21169774, 41457581, 41277924, 19030008, 43808533, 21150011, 44203158, 41357970, 41328220, 43198827, 40729275, 41334835, 35409339, 41328709, 43295293, 42939676, 40941085, 43816413, 41444846, 41332398, 41209375, 42939678, 36405756, 41222176, 43817675, 41211115, 41146908, 41339944, 43680041, 41120315, 41000998, 44037928, 35776184, 40833871, 41315396, 41331413, 41258031, 41102137, 41385752, 41333088, 41355632, 41335586, 43584778, 41216084, 21117206, 43837510, 41333338, 36783374, 41329905, 43142160, 36886828, 43602197, 40077886, 41133486, 41346543, 36056997, 41328197, 44111753, 40976986, 40990370, 41329624, 41005450, 41021304, 41309193, 21169769, 35787649, 41063226, 41354096, 41337102, 41178644, 41070712, 42726269, 41030402, 41198714, 43035004, 21169762, 41372001, 43806658, 43823903, 41334604, 40965876, 43842875, 43815602, 40898088, 41343363, 43854278, 43751769, 41336420, 36063796, 41353869, 36888679, 41445912, 40846155, 2030650, 40749466, 41328466, 41134803, 41337755, 40990350, 41008341, 41407864, 41321700, 41059501, 42482154, 19016867, 40729812, 41057544, 41333064, 41276147, 44162086, 21068028, 41216093, 35132472, 41161949, 36781656, 41357977, 41003381, 43517928, 21168389, 41218187, 35787134, 21094856, 40717874, 43654701, 41146909, 41330793, 35757656, 36880982, 41358625, 41022027, 40167745, 40903520, 40725714, 40171269, 41328918, 43835206, 41331969, 43765238, 40831537, 43169804, 40927197, 41339372, 41028251, 41279975, 21048436, 43618813, 2031940, 35141154, 41341531, 43751796, 41346544, 41337486, 35750567, 41353340, 43205921, 41328669, 40992707, 43799562, 41159959, 41332447, 44190173, 41338888, 41126087, 41412640, 41338700, 43268666, 41055988, 42934741, 43582464, 41387729, 40978233, 21119143, 41350943, 35159918, 40895064, 42934712, 21038538, 40926036, 43753203, 40884942, 43625769, 43807210, 40978241, 35865046, 35865670, 44161041, 42479575, 41451321, 21128920, 41052526, 40978239, 41182954, 41059460, 44084203, 35130480, 40234056, 43516361, 41307263, 21100969, 41334050, 41328173, 40836272, 41103354, 43682454, 41147173, 42725274, 41159936, 40987459, 40995327, 41128499, 35897671, 41334434, 21116401, 43146045, 36277138, 35776969, 43698026, 36278879, 41153746, 42942780, 43637547, 41353117, 42731182, 41359192, 41344502, 43141352, 41357831, 41329066, 43582432, 41331038, 21099590, 43608768, 41279984, 40234041, 43600509, 43690650, 21143934, 41337596, 21150010, 40717025, 41207178, 42942762, 21060303, 42725275, 41413230, 21079977, 41358507, 40102083, 41149172, 40822104, 43826082, 41028271, 43218482, 41331023, 41337484, 40718365, 41357510, 41216100, 36063790, 41120316, 43256487, 2055909, 40738836, 41102134, 41256885, 40729780, 41338188, 43662176, 41003396, 41358818, 43024877, 41334698, 40914667, 41247128, 41247138, 41331875, 43591334, 40853728, 40976981, 19107818, 41333354, 43024884, 41216096, 41009580, 41333834, 36054459, 21150009, 42939598, 41304568, 40738861, 41336720, 2057384, 1972873, 43823811, 41380746, 21109449, 41153757, 43198828, 41226018, 41412907, 43168181, 41094833, 35776453, 43211989, 40749532, 35779141, 41208727, 21138822, 41153706, 36277699, 43293983, 36810899, 44189814, 41003378, 40028865, 21155692, 43698024, 40885884, 43680042, 41335956, 40980297, 41336842, 41328778, 36073988, 41146917, 42716815, 41344533, 43751870, 41386956, 43144273, 43602196, 41304025, 43517815, 44098290, 21148661, 35745111, 40244439, 41332130, 41153731, 40073529, 41330744, 19067755, 41270545, 40853713, 41237666, 36884729, 43852579, 43619453, 41028259, 42934851, 43680141, 41277913, 43608747, 41331987, 41341908, 41332443, 35604740, 41354838, 40941053, 41191080, 40989737, 41367286, 36063788, 43583664, 40843480, 36278409, 41336739, 44190702, 41398233, 40839461, 41371873, 40915854, 41344287, 43196566, 43207307, 21065408, 43589818, 21070269, 2057302, 41336895, 41279993, 43636589, 41337999, 40234058, 41333725, 41393374, 41332343, 41277951, 41218177, 41030390, 40869016, 43783378, 41374862, 40231783, 41374986, 35897677, 41212679, 35787304, 35784821, 35776145, 41352847, 19079404, 44195367, 2055938, 41330382, 41333780, 21085073, 41059451, 41330546, 44059034, 44114047, 41360892, 41222184, 1550561, 41290390, 41345840, 43860221, 21168397, 35777089, 43816407, 21077939, 920598, 40965878, 43201195, 43860091, 41228486, 41363301, 41407865, 41011633, 41084127, 41330750, 40968020, 41399700, 41421401, 43289856, 41018423, 43662169, 35777837, 43679844, 41329498, 41328420, 36055498, 21126720, 41289146, 40903514, 41115346, 41344703, 43146046, 40738849, 40909696, 43257283, 41352281, 21169759, 21109379, 43728789, 35749318, 40233170, 40749558, 40738365, 21027993, 40825022, 41289145, 40738349, 40874704, 40234798, 40887053, 40872483, 44100653, 36054390, 41331194, 43662231, 41335790, 41359196, 41337636, 40965866, 41333784, 41182943, 40878712, 21158597, 43583665, 21136630, 41330657, 44115407, 40974061, 43737053, 1559864, 41353318, 21079971, 41420993, 41218193, 41334194, 42479078, 41356422, 42629019, 42482157, 41226021, 40738866, 41333340, 41054919, 40909711, 40855739, 41302314, 41030396, 41356515, 44084204, 41444839, 42939679, 19030001, 41335380, 41276169, 21128910, 41103366, 41153770, 41258030, 36785023, 2057278, 41393172, 43816445, 43662219, 43600462, 35758033, 41337703, 41330170, 43709700, 41340777, 35745115, 40964033, 41330638, 41103365, 41135697, 41389080, 42726076, 43142528, 41337187, 43835590, 41159950, 36783371, 41364878, 19067555, 41332135, 41065603, 21089674, 41329324, 40825955, 40903538, 40959854, 40862665, 41402140, 36888965, 43854531, 40874699, 41350895, 40911818, 41184746, 41337544, 41315379, 43854892, 43177427, 35775260, 43637545, 43745404, 41334433, 41329228, 43255457, 43853793, 41180289, 40821591, 40868139, 43613041, 43865720, 43589892, 43295299, 2031892, 41207627, 40173363, 41409687, 43726234, 41340263, 43024814, 21163691, 43592361, 42953125, 41329902, 41328919, 40167744, 41061518, 36277665, 40749562, 41329349, 41198715, 41218195, 43805689, 35865050, 19129511, 41240345, 40849562, 41327551, 41309148, 41149189, 21071596, 43716776, 41352531, 41337634, 41335500, 41331668, 41052521, 41336534, 41359679, 43157087, 1506315, 41388304, 40990360, 41249196, 40903534, 41342399, 35885559, 36879387, 41335962, 41304564, 41301633, 41388303, 35786410, 43620790, 43034995, 43667780, 36270914, 41352698, 41385099, 40905627, 43769756, 40821166, 41331043, 41028279, 41334423, 44125578, 43593512, 35770232, 41224144, 41412658, 41328195, 43190433, 40738321, 41331036, 40718377, 40870596, 41148597, 41454249, 40749547, 43582419, 43854030, 2919317, 21026171, 44073745, 41386364, 35886748, 41335377, 40897487, 40834514, 40843467, 43823813, 41414957, 41197418, 41328124, 40841340, 41122182, 43201207, 43146040, 41059462, 41309152, 41418826, 43745405, 43781652, 41329454, 35784800, 40749491, 43818806, 21126827, 41161937, 41335953, 40997247, 41331193, 41331202, 44099232, 43029932, 43264151, 43842731, 35777668, 35786756, 41271194, 44086008, 41330149, 21130235, 40886252, 43152652, 36888977, 36278296, 40965883, 35786937, 21104731, 42481958, 41331486, 43591217, 41333343, 21087011, 41320479, 36784236, 41331710, 42934864, 40847577, 40867944, 41059464, 2039828, 1559869, 2904164, 1550720, 40934715, 41333087, 36784242, 41018425, 41346174, 40841314, 41209379, 19131653, 41012567, 41136101, 43862193, 41357372, 40856719, 41342770, 40887117, 41328038, 21027989, 40872511, 41292455, 41353870, 35778513, 40929207, 43179245, 41331868, 40874705, 43769865, 41329736, 43734861, 35777836, 21135988, 41332168, 2039855, 40941084, 40962485, 40995318, 40968032, 41160385, 21168421, 21175333, 41339847, 41331913, 41114708, 41328419, 40877243, 41329554, 21060295, 43264138, 36421582, 36409014, 42934723, 41343533, 43763327, 21136375, 43175717, 41335999, 43040544, 21126721, 43806652, 43146036, 43286025, 43213647, 2034602, 41337801, 41184752, 40997195, 43708491, 41149500, 40990365, 40136334, 35770230, 41335047, 43164311, 21175335, 792429, 40909687, 41337756, 43619460, 35776417, 41333566, 41224139, 41328664, 41186936, 40927202, 21061654, 43852393, 41211698, 41342484, 44050423, 41359681, 41423947, 41178645, 41277943, 41369874, 43610352, 41287944, 40976974, 40991076, 41327909, 41444852, 41159939, 44196241, 40233205, 43133398, 41337705, 41026369, 41358916, 40878709, 43146330, 44058449, 40841295, 41327432, 44114494, 41361326, 43179242, 2057267, 42901997, 21070250, 19007008, 19101595, 43841985, 40964027, 43860240, 41333774, 21050480, 41329354, 41333250, 41184779, 41332344, 21099586, 35784324, 41329811, 41124293, 41336841, 41333767, 43179226, 41128518, 43201193, 40866339, 41338004, 40901666, 41354839, 41409152, 41196161, 35786662, 41093127, 41386054, 36035951, 41103350, 36781539, 43709441, 41328776, 41356423, 42934836, 41120296, 40970569, 41040663, 19024533, 43769849, 40997203, 19107858, 40749966, 41329449, 41352782, 41337368, 41328058, 21057910, 41357972, 41358745, 2057294, 2031948, 43190112, 21030813, 40976969, 21061659, 40941073, 43663050, 41341436, 41376098, 41240344, 41210054, 40895541, 44130609, 1972875, 41300506, 41146913, 43736045, 41340261, 44198300, 43262689, 43798084, 40749442, 44084205, 44076706, 41345722, 43841912, 43625792, 43770937, 21109439, 43630095, 42948511, 41210053, 40964025, 40943081, 41279998, 42482551, 35778790, 41333074, 43591951, 40914649, 44184310, 43724659, 42902110, 43654649, 41128503, 43188407, 2031950, 41271183, 42934749, 41279995, 40738337, 41147607, 40852488, 40738858, 42934715, 40944874, 35777187, 19108490, 40957291, 41327751, 36073740, 21120468, 41333944, 41253163, 36896300, 40715934, 42714567, 43028391, 41444850, 41209367, 41176066, 41336819, 42922452, 41335098, 43628443, 43516593, 40959851, 41222213, 41330010, 41245196, 41332550, 41317521, 40980296, 41328171, 41039419, 2057319, 43702562, 43726778, 41357116, 44195831, 41364105, 40107135, 41118747, 41328835, 41030401, 41355568, 40980300, 43041650, 21030799, 43659280, 41331027, 41197426, 42873629, 41408983, 21050482, 41155867, 41327788, 35408671, 41343145, 41340142, 44188412, 41329067, 41334697, 41292460, 41117719, 40980486, 43295298, 21087584, 21057547, 41199511, 41335276, 41412600, 41124308, 41330169, 43602203, 41270548, 40738358, 41260197, 43663685, 40847587, 40749453, 35757662, 41340948, 41345721, 40927203, 41378473, 41366388, 35150552, 43847635, 43698178, 40841332, 43807883, 44057351, 41394106, 43733937, 43844068, 41309189, 43770745, 41337405, 43715863, 41260566, 41227225, 41003383, 41100982, 41009585, 40241180, 41302299, 43260752, 40749492, 41364902, 36058282, 41249188, 43672816, 41240116, 43191813, 40821694, 40852482, 41122194, 43843369, 42934819, 35754363, 43201191, 40862773, 36417902, 1592240, 41038258, 41197417, 35759414, 36781246, 43517883, 40895294, 41472837, 41052530, 35852501, 40171268, 41009591, 35143508, 43681438, 41216107, 40717024, 41345131, 21143924, 36278194, 41367418, 43199420, 19129254, 41334051, 41360130, 41020693, 35779142, 19106694, 43257773, 43700133, 41128493, 43744496, 40938649, 40234034, 43780473, 43841438, 36781269, 1518292, 41343470, 41353316, 40914655, 41328613, 43737800, 40923904, 41184735, 41286186, 44189457, 43708825, 40862764, 21163620, 42724497, 43638935, 40832669, 44047932, 40972014, 41051258, 41335562, 40732463, 21159969, 21116291, 41074059, 42629017, 41009595, 42922454, 40749471, 43201187, 44174547, 40738328, 2057259, 41336673, 44180581, 43591934, 36054462, 41361113, 41424286, 21165614, 41247115, 40990361, 41327758, 40731082, 40958450, 35851457, 43852580, 19021571, 41059492, 43625926, 40234045, 41337108, 41356941, 40738413, 43038973, 41122212, 41117717, 41335565, 44064631, 40946044, 21148648, 43699279, 43626775, 41115356, 36419766, 41327446, 41134790, 21021097, 41222200, 40837882, 41336529, 42934838, 41008354, 40883731, 41388917, 40930143, 2057264, 41216064, 41286183, 43781738, 41445858, 19063770, 41334714, 40907295, 43261325, 41216053, 41412534, 19019893, 36779830, 43773711, 41335151, 44063075, 43787721, 41330986, 41338012, 40900082, 41336978, 41357526, 43834955, 43590656, 41368640, 19029656, 42720886, 40936910, 19107560, 40018868, 40947267, 43772039, 40839462, 43809499, 41028265, 43197641, 21120467, 41340333, 21070247, 40932763, 41330980, 21158601, 41327749, 41334373, 43788567, 43787851, 43817580, 43583873, 40872514, 41008347, 41091047, 41337520, 41412735, 41177212, 40866340, 40233950, 41227224, 21060298, 41020697, 41336723, 41410203, 41026374, 36054450, 21030827, 43753202, 43637122, 21037912, 35780319, 43752701, 41070720, 41279997, 42941637, 36892649, 40948171, 42902158, 40749446, 40073808, 1551234, 35787698, 43672766, 21091046, 43698956, 36073739, 36277702, 40867958, 43157294, 21087352, 36277524, 41214269, 41335095, 35885164, 41126997, 43697948, 43644038, 40972039, 43658486, 36886911, 41329321, 43745402, 43733931, 43726442, 43736367, 21061676, 21109435, 43190113, 41336239, 43691569, 41330379, 19031623, 41329350, 41193136, 41026373, 41357640, 41222192, 41333570, 41284161, 41290391, 43177432, 40848461, 35750571, 40947249, 43179035, 41339080, 43734869, 43771279, 41128474, 43751750, 36272969, 41329765, 44190527, 2042112, 41135698, 41136907, 41309153, 41367417, 41005446, 41345129, 1518491, 43715829, 43823898, 40145461, 41133493, 41359195, 41358820, 44185763, 41261486, 1550775, 40146415, 41182944, 43733975, 41155868, 43853794, 43762893, 41026358, 41224168, 2054982, 41321695, 41354308, 41419040, 41054923, 40909681, 40867951, 41061513, 21094848, 41153758, 41337357, 41356161, 35604852, 41009589, 43218481, 41133479, 41361157, 43644328, 41340220, 40974062, 19129398, 41115727, 41425891, 35776908, 40961455, 793639, 21176022, 2910254, 35606555, 41395960, 41337714, 2057280, 43709442, 43716047, 35865044, 2012261, 44036919, 21158609, 21051865, 21029488, 40997211, 42939647, 41427297, 43031871, 41102136, 40903492, 21099572, 40903506, 43638302, 44124685, 41227241, 41161933, 41193130, 21022449, 19067190, 43708486, 41026366, 41302320, 1719023, 43745024, 41333008, 43746704, 41329453, 40725704, 41122203, 40718364, 43772665, 44189369, 43825237, 40738843, 40887051, 41328411, 43801574, 21168404, 41087458, 41331051, 41407798, 35778349, 40872494, 36260706, 41166958, 21032177, 1150807, 43291358, 19104629, 19039743, 41218181, 41417592, 41209166, 40717163, 43255459, 44050424, 41336486, 40972030, 43583431, 40234047, 40925592, 43852610, 41337262, 41346433, 40964019, 43780795, 41343608, 43196440, 41336211, 40907301, 41281734, 41197420, 920595, 41367749, 41336483, 41186920, 43600173, 41309166, 43727427, 40911815, 41335503, 35897729, 41344624, 41001035, 1551192, 43196464, 41345972, 41336896, 21109452, 41303656, 44164822, 41091040, 40938635, 43805768, 41028264, 43595369, 40896827, 42942783, 35778429, 43517846, 36420563, 43762054, 40901673, 43755518, 41309172, 41328027, 40965881, 35777111, 41028270, 41059485, 41359802, 41338229, 41028240, 19034805, 41153721, 43672763, 41329815, 41334049, 21127484, 40887056, 43179229, 41239691, 43141439, 44198238, 41070700, 44049595, 36119166, 41352743, 43698043, 40997237, 40849569, 21120473, 41177855, 41375469, 41327917, 19018083, 40234101, 2057311, 41364568, 41412866, 40738338, 21089675, 40931201, 41337408, 41028275, 41095742, 42948557, 41018421, 43157322, 41091080, 19135275, 41329999, 43175718, 41359540, 41336465, 41070703, 1518525, 41271857, 36272970, 40738340, 43798409, 41093122, 964312, 41122217, 41333023, 41339163, 40995322, 21081309, 35828575, 41023969, 41149176, 40978244, 40903535, 41323858, 41340262, 41315397, 41340449, 41328427, 40154981, 21155687, 43852618, 41386594, 42948516, 41158428, 41311330, 36505168, 41052529, 41247094, 40729817, 43152624, 41330639, 1832196, 1506479, 41209383, 41331045, 41337104, 43700233, 43201204, 43168183, 41009578, 36404420, 36035954, 43762333, 42941642, 36895095, 41206255, 43157284, 40883723, 36054448, 41356164, 40028599, 40959847, 2057321, 21167580, 41071930, 21099521, 41331094, 42481957, 40909714, 40847593, 40915858, 41328421, 40955918, 43708487, 2034605, 41335050, 41196163, 41329373, 41365492, 42939644, 43034993, 41445666, 41315393, 35886741, 21022441, 43152579, 41074263, 35897664, 40909706, 41222189, 41164812, 41084116, 40870595, 21124123, 21068029, 41193128, 41327564, 40872477, 41256887, 40865642, 43771325, 41337006, 40927204, 41331323, 43289862, 41333073, 35852508, 40085517, 41451316, 43135458, 41330126, 35778347, 40841293, 41237745, 41354040, 41333011, 41342546, 1510435, 43830498, 43662171, 41186940, 40824529, 41344601, 41333781, 41385223, 41186934, 21168413, 43171060, 40898087, 2055924, 41334692, 2057395, 40932774, 21089682, 793636, 43680966, 41028285, 41337085, 41331706, 19128780, 41040669, 35408963, 41114019, 40903549, 41290398, 43751770, 41333346, 41332558, 41425479, 44168677, 793637, 40749556, 35411846, 40946030, 40997248, 41338164, 41228464, 41196151, 36035956, 42948565, 40085504, 40964038, 35134890, 41330745, 43734870, 41053289, 41333254, 41084735, 43584780, 41332665, 41061521, 41186928, 43591932, 41357750, 35787135, 40738865, 43789074, 40960517, 41336050, 43664659, 41019526, 36036034, 21071592, 41343472, 41159951, 41409383, 41337139, 43662085, 36063381, 40908260, 41028282, 41353866, 43715756, 19033379, 41333498, 36277867, 36073745, 19019891, 41329347, 43133389, 43646848, 41327739, 40941089, 44135638, 44108950, 40738324, 43716549, 41055975, 21145865, 21059465, 40995298, 41153718, 43219507, 40716466, 40725711, 1971535, 36260705, 43600454, 43196567, 41458096, 41334832, 43636928, 41386291, 40883732, 41407436, 43584783, 41329702, 43805798, 41065636, 41364634, 41146904, 21050469, 41191091, 41332449, 35779018, 41333012, 35606554, 44172487, 19128896, 43853849, 43201180, 41307260, 35161171, 41290397, 21081312, 44112415, 41350958, 41175057, 44063429, 43751828, 40938670, 41353230, 40749476, 41032091, 41017529, 41337373, 44124382, 21138806, 41253140, 41338628, 40936902, 43751768, 43152619, 43781399, 21032178, 43584167, 36277704, 43219432, 43190435, 43644118, 43190429, 43698177, 35897679, 41242613, 41444848, 44173121, 40896810, 40027452, 41385818, 41153719, 41122198, 43716696, 43589544, 41351340, 41268126, 41332010, 40865646, 43201203, 40959478, 35776907, 21085080, 41159929, 40999352, 44088623, 43164740, 36892650, 21089696, 41105498, 41342102, 44190507, 21070268, 43769515, 43582469, 41331483, 21077274, 40956862, 43667671, 35830397, 1551044, 41336210, 41059494, 41292456, 42902114, 41445857, 40905632, 41091066, 41359806, 41329903, 44130046, 41385264, 36895121, 41309151, 41407801, 41335273, 40972021, 43763787, 41009584, 41089073, 44097107, 40969686, 42922447, 41386595, 40729788, 40824948, 40909680, 21030821, 43841897, 43039462, 21156722, 43727420, 40883722, 42934717, 41329901, 41052533, 43034977, 42728936, 43600449, 40883735, 40867959, 43662179, 41334375, 21070272, 41333525, 41159942, 41177875, 35144764, 44050035, 41230586, 41337354, 41355064, 41373093, 41226020, 41327913, 40085513, 40738841, 40845223, 43042652, 19009120, 41332663, 41115364, 42901998, 41343409, 41361501, 41332846, 41330381, 41151893, 41081185, 40932771, 43834626, 1510782, 42953126, 40866953, 41413093, 21050476, 43610551, 43690615, 43734855, 2057255, 36780399, 41407725, 41329057, 43733965, 40961639, 41426459, 41209368, 43179227, 41327741, 43663056, 40947257, 21165500, 41059489, 40915871, 41328710, 43629651, 43806655, 21165618, 41065634, 36063363, 43754641, 44098962, 41359589, 44035929, 35776527, 40060728, 41345073, 21040698, 41070715, 41359178, 41359500, 41385222, 43607913, 19038300, 21079959, 41093130, 41210052, 43691570, 40233962, 40732455, 43751767, 35865669, 41228480, 40903553, 21120470, 42934876, 40993733, 41084128, 40927974, 41336345, 41309179, 41358885, 42922449, 43798366, 43864774, 43737574, 40901672, 40936907, 41460018, 2055925, 42934827, 43257278, 41387957, 43142531, 41356307, 41342977, 41339233, 2057366, 41353233, 21148669, 36879342, 41335749, 41345632, 35138513, 21140174, 41104684, 40872501, 43041655, 36036038, 41345909, 44164907, 43716697, 42948510, 41061510, 43817859, 41332333, 41412580, 40853718, 41330981, 43197552, 40932759, 43180858, 40941072, 41351860, 41059495, 41412733, 35157100, 21061650, 41050800, 21140172, 40716464, 35851450, 43638938, 21030829, 40976980, 41149171, 35753898, 36063370, 40997218, 41153735, 41305682, 43824811, 43751748, 41079961, 36812136, 43663059, 36119167, 40915850, 41177864, 41255135, 41120314, 40073457, 44198508, 40957793, 40749441, 43734172, 43716741, 41091048, 41337105, 43637544, 43146328, 21124062, 41093119, 41337140, 36074595, 21107264, 40874706, 41253154, 41333732, 43514675, 42480349, 41346113, 43201202, 43698953, 43190431, 43733939, 1559867, 41216077, 43645419, 41336870, 43716690, 43155454, 21067383, 41329450, 41245202, 2034706, 41352161, 35759416, 35786218, 43823900, 35761826, 41222202, 41320486, 41409883, 41358679, 21159966, 43625869, 43789075, 43185408, 41480900, 2034601, 21078119, 36063801, 35606596, 41153711, 35897728, 41357635, 43275322, 43862337, 41332082, 1518261, 40234030, 41327717, 21039468, 41330021, 21158596, 41216078, 41346423, 43672765, 43691722, 44076133, 43818803, 40914666, 43208129, 1150806, 43518126, 41407198, 21098213, 42939685, 41332327, 41028274, 43638940, 43697949, 40846154, 41115363, 41351413, 41336480, 41344938, 35777297, 43627277, 41284155, 41385831, 44197823, 41335083, 40738832, 41455896, 43518029, 41304538, 41164813, 41021494, 41309196, 40749458, 41328030, 41317512, 37003557, 41409060, 40995315, 41243664, 35151761, 21089687, 40749493, 21169771, 41334381, 41352848, 43769850, 41276149, 41216052, 41338317, 21109445, 41091057, 35142848, 41070704, 41136906, 1551045, 43035007, 41355999, 41177214, 40965861, 41039432, 589198, 40832745, 41340143, 41304024, 40997205, 21057909, 43654694, 43762320, 43135454, 21030824, 40904042, 19061583, 41461344, 41320482, 43745400, 40073530, 41332404, 41061509, 2034696, 43201197, 43755525, 43788575, 36891842, 43787732, 2030651, 43197226, 35134736, 41159935, 41415345, 41337711, 36056986, 41327762, 41239687, 43774416, 41328170, 40738833, 41276151, 40717021, 41327406, 43654349, 36809976, 41387581, 40749436, 41336207, 41354929, 36055502, 35780433, 35142272, 41392245, 36055508, 41113539, 43031878, 40822413, 44076704, 2039820, 44191522, 41307238, 43582463, 2057258, 40884956, 41337482, 42722632, 41292627, 41247103, 41330791, 40965884, 2034617, 41070693, 43788633, 21155693, 43690608, 43201489, 43793204, 40958960, 42922387, 43816771, 40961463, 41247140, 40122517, 41180287, 41328579, 35765909, 41346422, 43584779, 43190118, 41331608, 43798371, 21116402, 41344047, 41336747, 41392722, 41390827, 41327923, 21070223, 41338699, 40841331, 41339613, 42942785, 40931205, 44063072, 42479557, 43816410, 42934711, 19006962, 40999354, 40717802, 44130285, 43715864, 2055010, 36063798, 41052525, 40965858, 36055495, 41329010, 35753486, 43190135, 40872034, 41065623, 41003377, 43619722, 41366843, 40987375, 40903508, 41416330, 44060313, 40122520, 43590790, 43788586, 43787726, 41334636, 21040710, 40095885, 21146473, 41341652, 35750570, 21089671, 43817389, 35745122, 41329270, 36780476, 41157581, 41358189, 41087463, 43842184, 36880934, 40999372, 43266789, 41253144, 40930676, 41337356, 40987458, 19063762, 21058267, 43583432, 41327510, 43190114, 41065617, 21021111, 41331411, 41331661, 41240506, 41166062, 41153761, 40914651, 41332085, 41388024, 43853851, 43845654, 21169775, 21116759, 40932764, 41352283, 41328580, 41352928, 41026359, 43267300, 43852570, 21061681, 21116755, 41150248, 36063782, 36893668, 40852465, 40947262, 43698074, 40997202, 41351729, 41030399, 41311331, 36074596, 21166470, 41332402, 43780463, 43692523, 41337103, 42934740, 41328998, 41346680, 41289130, 44187517, 41159928, 40941065, 40975827, 43284462, 41030395, 41451313, 40992704, 41344074, 43296751, 21128908, 41328378, 43193885, 43697439, 41071919, 43706727, 21148662, 43146048, 41184768, 41359011, 41345571, 40997221, 1592243, 19067729, 21047744, 41315411, 43157323, 41329231, 40878659, 43275186, 40980302, 42934872, 40718363, 19030005, 41342049, 43135456, 41337098, 35780546, 41338701, 41331873, 42717310, 21168414, 40928508, 41317497, 21119139, 41337100, 42876079, 41333937, 42934874, 19007007, 42729023, 41133491, 41338936, 41336531, 40738851, 41177858, 35885740, 40234806, 41122171, 43751832, 41120310, 41128491, 21075346, 41332657, 43157329, 2057238, 44086625, 40901667, 41330790, 41357417, 36888978, 43583433, 41385911, 2042113, 40903505, 41327437, 35776073, 2030886, 43619449, 43842022, 41186939, 43154061, 41143710, 44176891, 41159955, 40914653, 41355002, 41356303, 41228484, 40903523, 41057546, 44049596, 41179282, 41239695, 44050037, 41255163, 43681528, 40959121, 41340720, 43769847, 41337099, 43680061, 41277958, 43770064, 2057304, 41339373, 35786998, 43590807, 40085501, 41336053, 41392920, 40976996, 43727712, 41337005, 36073741, 43824817, 36781540, 43582420, 21094862, 41211696, 41346547, 40991078, 44189759, 41352674, 19133854, 21061668, 43136803, 43211788, 41337142, 40738357, 41034499, 43715827, 41020692, 40947260, 43730212, 40989735, 43655961, 41126995, 41301632, 43673358, 35781046, 43791571, 43841986, 41335149, 40903537, 41034769, 43135446, 41328425, 41228488, 43655902, 35131073, 36421585, 40863869, 43135169, 41345903, 41247126, 43627251, 41184758, 43024879, 41328006, 44075787, 41363425, 21145719, 43847634, 41179281, 41009581, 41331508, 43157291, 35144774, 43024874, 2055919, 40725263, 41332166, 43860172, 41330936, 43823905, 21050459, 21165501, 40073190, 41360037, 36506879, 41228481, 41329760, 41328369, 37499312, 43716687, 43028402, 2057252, 41277928, 41451323, 21138823, 43644136, 40908262, 41052851, 19002013, 40903545, 21060316, 43842881, 41330650, 43842884, 35780024, 40927821, 40964022, 41071923, 44102604, 40964030, 40995325, 41328422, 41415167, 43517836, 40959111, 41414843, 41338316, 35161252, 40896987, 19104623, 41070710, 35885557, 40729814, 40018869, 40989734, 44195515, 35830462, 2057303, 43031873, 40949296, 41071918, 21021120, 19034648, 41327921, 42729022, 41332165, 35151805, 41153774, 44057349, 41358362, 40843472, 41115347, 40843477, 40909720, 43733861, 21150020, 41133483, 42902108, 41451330, 43190131, 43263177, 2057268, 21030768, 41359588, 40927820, 40965905, 41182935, 41270552, 2034598, 36880204, 40962471, 40821715, 41358163, 41253166, 41328418, 41357614, 41370353, 1831859, 41451320, 41342326, 41333072, 41116098, 41337740, 42948553, 40824931, 40964032, 41298595, 41300969, 41351399, 41049718, 41339701, 19034650, 41401317, 21061658, 40085187, 41399639, 43751961, 43133395, 43257282, 41343433, 40927813, 41242022, 35776901, 41366844, 40934703, 43595537, 43034992, 43034987, 41334242, 41335091, 41337355, 41003384, 41059466, 43662174, 41184738, 40880788, 43517800, 43755723, 41249198, 41328122, 41388903, 41331048, 40049686, 41353560, 41018424, 41122224, 43268147, 43168172, 40738844, 41328818, 41339869, 41336736, 36279232, 41334414, 43028390, 42479574, 43698277, 40968036, 43589820, 41307257, 41151892, 41091577, 41194982, 35778885, 41146923, 40872537, 41337545, 41358331, 40915869, 41174955, 41451326, 41332384, 41164809, 41333242, 43727421, 41218182, 41409140, 21045747, 41097155, 40717020, 41332870, 43805711, 42731325, 41293441, 21087013, 41147608, 40834501, 43709921, 41336719, 44195886, 43295803, 41334031, 21071602, 43174554, 43582774, 41097240, 41335663, 41030394, 40946024, 44192344, 19019894, 41276161, 41327426, 41034516, 41339167, 43135448, 41328415, 43676556, 41408585, 43860422, 43668434, 36784706, 40946031, 41120313, 40870602, 41124292, 43646124, 2057323, 43697932, 41346175, 41216073, 44136548, 42730381, 43681472, 43787728, 41315412, 40903503, 41328711, 41147609, 36118608, 40852472, 21081302, 1719046, 43672818, 41392916, 43526137, 21070267, 40958453, 43179240, 41357321, 43190127, 41311333, 40892759, 41328377, 36054464, 41309184, 41276155, 41353676, 40841320, 41329346, 21140168, 40972012, 41309140, 43845630, 41028276, 41335384, 41083986, 44177144, 36277266, 41360201, 41358908, 43638024, 35775696, 41084861, 35779752, 40073531, 40931195, 41184769, 43152444, 40903536, 43780464, 42942756, 41461972, 40732460, 36063379, 21061666, 41180281, 35150827, 35763617, 19105671, 41158434, 41332777, 21040697, 41093126, 41053985, 19121838, 41155861, 44190464, 41277948, 43862244, 2012263, 21124125, 40880787, 41333364, 43765544, 40988102, 2034607, 43852611, 40956197, 43655631, 43190117, 43716699, 40932773, 43842879, 41397867, 41407845, 41389439, 41182942, 42942764, 41339052, 41028262, 43781408, 21116290, 43517847, 40835878, 41182939, 43296719, 41161935, 41334225, 40965910, 43278270, 44076707, 36506027, 41355468, 41259303, 41091081, 21077277, 43141515, 43199413, 43289858, 40965892, 41350834, 36063778, 43152678, 42934853, 43746703, 44111752, 41305681, 42939645, 21143922, 1518851, 41346540, 40976995, 40992705, 43787729, 36420020, 19114216, 40872533, 41328666, 43842883, 41024722, 41341435, 40965899, 41336490, 41222195, 41354822, 40897484, 41097163, 41147610, 41341470, 21109448, 41305102, 42948514, 40934761, 35145933, 41330976, 41340448, 41358363, 36895130, 40934718, 44096035, 35886742, 41341868, 35758466, 35780038, 41330642, 41071922, 41330125, 41333024, 41329754, 40934706, 21067384, 43592083, 41336630, 43028389, 44165136, 43854277, 41084122, 2031972, 40972016, 41315384, 43709047, 2910252, 43823910, 2057310, 43763663, 41117714, 41115369, 36054451, 21060309, 41418204, 21084959, 41185080, 36880017, 41216097, 41336071, 41336299, 21138808, 41102126, 43135170, 41200475, 41333075, 43602200, 43709920, 40965911, 41355677, 40729272, 41344095, 43264166, 41355521, 43644962, 43826010, 40872495, 41337137, 40122521, 41335881, 41216104, 40928511, 40991756, 41034507, 36895073, 40749467, 35886728, 35157498, 41209371, 41084115, 41042731, 35897726, 43807209, 41166072, 43042584, 44064008, 41166068, 41184739, 41457857, 41102119, 41329069, 41182946, 41335110, 40896814, 19130270, 41343266, 41159960, 36272972, 36781247, 41177213, 41222194, 41122207, 41342752, 41053291, 40926035, 44166105, 41309171, 42934850, 41417507, 41331028, 40872475, 43748133, 41333341, 40725717, 41009594, 41019527, 43286026, 41335998, 35746406, 43644958, 41083478, 40738848, 40821728, 41333941, 41344810, 35749966, 43663549, 19067758, 41228476, 41137102, 43656425, 41216108, 41379327, 40941063, 41481277, 41227236, 43590896, 41337259, 41334696, 41332682, 36063374, 41097159, 41341692, 41332392, 41331712, 41331040, 975169, 19018084, 41334602, 41328423, 44190489, 41330387, 43823958, 21035955, 36064314, 36420018, 21100972, 43211780, 41358681, 35758285, 41336482, 41327401, 42939683, 42934738, 41309173, 41008351, 41456330, 43769888, 41340719, 21138796, 43674796, 42948531, 41053294, 36280086, 35753474, 35776144, 35865049, 43164691, 40990351, 19009117, 40948532, 41093123, 43773637, 43843388, 41329495, 43662180, 42480362, 40841292, 43179536, 40234099, 40884962, 41342401, 40020968, 43681466, 41332163, 44181057, 35751050, 21173456, 41115362, 2057291, 40018149, 40738837, 43825342, 36262173, 43736647, 40866349, 21096669, 41333296, 42629021, 21108386, 36408719, 40738868, 21060279, 41357616, 35777603, 40959849, 42934824, 43770742, 43790193, 41230583, 42939641, 41008343, 41149186, 43798407, 35776633, 21124134, 41329056, 43211990, 41328370, 40849577, 40909710, 41317494, 41329108, 41301631, 40934743, 40234031, 40932772, 44074140, 41323808, 41216080, 41329054, 43601722, 41332678, 41315401, 41344602, 41336527, 40738346, 35776236, 41253150, 41378808, 41412895, 40231770, 35154389, 41115374, 40870586, 44125584, 41335563, 40854569, 43168156, 41417921, 43646810, 35131353, 43665019, 41328408, 35766596, 41097173, 21068777, 40738351, 43272793, 1972856, 40903493, 2039826, 41407318, 41309139, 43780462, 41022684, 41330751, 41359882, 43028404, 40936915, 41329111, 43692014, 41030389, 41339559, 35784819, 41335964, 41330124, 21030800, 40947269, 44101648, 40986252, 40997206, 44202401, 41336748, 36510096, 2057368, 41365165, 41186918, 41343534, 41334384, 21145718, 43031877, 41335410, 41339614, 904511, 41339558, 41482738, 41409382, 41303659, 43769903, 41337377, 43157286, 35159807, 21169768, 41084133, 43038700, 42941656, 41026370, 792331, 40965860, 41335785, 41207626, 41091400, 21032176, 43584781, 44195579, 36781267, 41143086, 43619038, 40901665, 41332559, 40717026, 41336055, 41418146, 41335960, 43752697, 40862771, 43676053, 40941095, 41114704, 41327539, 41302316, 21051861, 41122175, 40836816, 43852612, 43166400, 41153727, 41126996, 36264752, 41115354, 41112152, 41331497, 1150862, 41360129, 41251800, 44159801, 19027188, 41315391, 41021335, 21138778, 40932769, 41186916, 41335408, 41346138, 41372402, 41052520, 41345130, 41355317, 41212672, 40825175, 44174817, 41337135, 41356963, 41338003, 44098961, 19006961, 41329957, 41345507, 44062059, 43646581, 1971536, 36063777, 41329227, 41397258, 35141230, 41331205, 35783630, 43783372, 40847578, 41339893, 40896813, 41342400, 2031877, 41218192, 40073204, 36893742, 43771228, 43614133, 40884959, 21106629, 41271201, 40903524, 41239692, 43763474, 41080439, 41054905, 40968038, 41342402, 43608765, 41364249, 41327924, 41333450, 41003370, 43663157, 41118746, 43589807, 41186931, 41407435, 40997196, 43716777, 40055572, 21099612, 42873627, 41329005, 41346665, 41332386, 43733952, 41328403, 41402553, 41407516, 21057423, 43607911, 41091805, 41064167, 41355063, 43601931, 43619723, 41153274, 41352282, 41122221, 21138811, 41332681, 41334032, 41097154, 40847604, 40019100, 21128916, 43152443, 41282645, 41103357, 41020679, 36055493, 41212688, 43034994, 40934758, 41091068, 43211976, 40958451, 43823906, 41336263, 43825178, 41328372, 40846120, 44045824, 36277847, 41327793, 44035928, 35786828, 44186081, 41001937, 19061425, 43682125, 41118755, 43806661, 40853721, 41009599, 40822536, 36894466, 41338215, 41339698, 41344985, 41177852, 43211786, 41292461, 2034619, 43157575, 43582989, 41028287, 2030656, 41330219, 41332390, 43655202, 41337406, 19006939, 40130114, 43692874, 40909718, 40968039, 42934718, 41122202, 43264139, 2057315, 21163685, 41339165, 43787725, 35851456, 2034702, 41384895, 41122206, 41247087, 40867954, 41116727, 40872529, 43211986, 43697946, 43680970, 41302315, 41364635, 40834505, 1507735, 41023671, 21114520, 43517799, 41259306, 35776072, 41039434, 41168092, 41363500, 40934760, 43715828, 41337523, 43636636, 36418764, 43680247, 41346363, 41222178, 21158602, 41049712, 21136382, 36063376, 36263907, 41412622, 41345720, 41336187, 41333071, 36055507, 36278748, 44112414, 40749465, 41102139, 41368019, 19103651, 40965908, 44049036, 40749464, 41059496, 40824287, 40872503, 41332847, 21156432, 43683927, 41091036, 41126080, 42479559, 43163559, 21109443, 41343783, 36885712, 920718, 41127021, 41216102, 40972036, 41182952, 40872498, 41284149, 41352593, 41332676, 41166054, 41332328, 41286168, 42942782, 43726407, 41091037, 41340177, 43600819, 41132325, 41384925, 43034974, 43291491, 40960512, 41358819, 35776842, 40171267, 21051868, 41040657, 40853722, 43691572, 41222179, 41309190, 40911806, 40725712, 19001446, 40716468, 42934866, 40749433, 41330654, 43175643, 40241185, 40872850, 41320481, 41362387, 41455496, 40711225, 43518087, 41149174, 41358911, 42934709, 41335092, 41041545, 36419763, 2027980, 19106203, 43755476, 41451322, 43038699, 41330748, 41344096, 40936899, 43038701, 21134021, 41120304, 41276156, 2057367, 41327411, 43790861, 40990363, 41227223, 41184766, 21061678, 41333078, 21060311, 41231524, 41379520, 43590804, 21048130, 19050909, 40947252, 41358680, 21109434, 40997243, 40897491, 19129069, 43197553, 41334908, 43717230, 41122186, 43826080, 41336817, 36056992, 41211694, 41335378, 43654634, 35786758, 1972876, 41124297, 41305676, 43629879, 41451327, 40997228, 43034998, 21089622, 41320487, 40903498, 41021301, 40836811, 41336467, 41356398, 41330391, 41307250, 19030003, 40749443, 21140177, 40896817, 41334380, 41085522, 21158590, 40964014, 41357973, 35140962, 41407800, 41363908, 43289866, 41159933, 21026178, 21077664, 41292576, 43583422, 41445867, 40956195, 35142027, 41352449, 43188395, 41277912, 41184784, 43805687, 41302307, 43040143, 41336625, 2034632, 21145862, 41302297, 41423283, 43841989, 43600477, 43860174, 43212160, 36882257, 21109437, 41333478, 40738356, 41199516, 36885757, 41105493, 40738331, 40991742, 41250884, 41337635, 41344739, 41040662, 2042110, 21119135, 41287942, 35886740, 41333069, 40073202, 41241040, 41087467, 43280552, 42934840, 41151889, 41359401, 41253162, 41329910, 2024031, 41034510, 42948560, 43771203, 41331988, 43646523, 41340643, 40838026, 41336479, 40930149, 36265457, 41083487, 42921165, 40855863, 42934828, 43218548, 1551008, 35777604, 40903526, 41426961, 43656430, 41317493, 40865007, 43210222, 43698948, 2027979, 40873338, 1592251, 41258024, 41335564, 43636299, 41363126, 41329230, 41331348, 43708517, 35136287, 41332610, 40749489, 2057265, 40835203, 21096672, 44190509, 41386363, 41273503, 44196097, 43835585, 41315385, 41330002, 35776418, 41354931, 42918694, 42948556, 41274528, 41327908, 41218179, 40903504, 43287584, 43663462, 41289131, 41334702, 43179528, 41335006, 43202832, 41343364, 41008339, 35159631, 41332611, 42724498, 41412950, 41336374, 41321703, 43644955, 41331037, 41337090, 40749974, 36277909, 36780397, 43680139, 40997226, 40729781, 40968035, 21068238, 43179534, 1551010, 41115355, 43781559, 21104724, 35776912, 41177856, 40969704, 41332345, 40990357, 41346176, 43626768, 43024881, 1506430, 44047020, 43862338, 35751048, 41344426, 43146038, 40849563, 41358190, 41329524, 36056980, 2057317, 41256884, 41321707, 41386968, 21136831, 41352269, 42479774, 36885755, 36063365, 44047295, 40883738, 35784449, 44190572, 40903515, 43770063, 40997204, 41335808, 41355679, 41340271, 43824809, 43262690, 19113791, 36279274, 36063795, 43619452, 43516065, 40073191, 41354952, 41174952, 21158613, 43656422, 41341207, 42723870, 41334422, 41186927, 41184750, 41206551, 43848226, 41089086, 1719013, 41363843, 42481445, 43789772, 41324807, 920717, 41330990, 42939639, 36887713, 36781613, 41331025, 43691578, 44173776, 41262275, 41409384, 21067262, 43680045, 41323806, 41344656, 41247109, 21060278, 42934871, 36888942, 43135152, 40073192, 41259308, 36055503, 41153771, 35830350, 41052852, 40911811, 35758029, 41040665, 40141448, 44035264, 40725281, 1972356, 21163684, 41228483, 19121385, 42941635, 43802658, 40964026, 40959725, 41122185, 21099576, 43752703, 43146024, 21061667, 40738360, 43824796, 41331754, 40959398, 43589817, 43781409, 40884961, 41122204, 40997199, 35141914, 35749963, 43041653, 41354039, 41354951, 21089672, 43645261, 43764029, 920716, 42934729, 21081313, 41413009, 43752044, 41359883, 40122519, 41307252, 41363503, 21148646, 41329051, 40878678, 41344288, 21037914, 43824797, 2057281, 41331033, 21078101, 43747262, 41366480, 41355137, 36891672, 35777562, 41327757, 40234102, 43271750, 43831628, 40718372, 40943083, 41418827, 41067727, 43265788, 40909715, 41245176, 43177435, 43196614, 43584168, 43608764, 41059490, 41333010, 41426547, 41331289, 41191085, 41329048, 19034232, 41085526, 41277946, 40928513, 40955919, 43024867, 43637145, 43218310, 41168091, 41184767, 41271181, 41299748, 40872517, 36896040, 41114707, 41344813, 40234049, 41337522, 41482737, 40872521, 41196150, 41216094, 43589932, 35781187, 1506426, 41091093, 41335584, 43210223, 41331298, 41335745, 975193, 36896522, 41342181, 41329003, 40717877, 41329001, 41337134, 40972029, 40959100, 40872548, 41289141, 41327715, 43153590, 41179676, 40866342, 41030392, 41331044, 41364567, 41334420, 41333836, 2034624, 41413023, 41228467, 41359074, 35777565, 43601425, 41273501, 41005428, 41276165, 40738334, 41331990, 43698021, 40896818, 41186930, 41334055, 43211999, 36064313, 21157598, 43735451, 41351380, 21107249, 41218189, 40835200, 43753249, 40989728, 41333567, 43219875, 41104683, 40824837, 41329759, 36026947, 41332551, 43592084, 41023668, 41334385, 43680179, 41339919, 40085179, 40234791, 40930151, 40839459, 44184562, 1551042, 40947266, 21148652, 41331293, 44074757, 43826974, 43610353, 41354602, 19067791, 40847585, 41358774, 40848027, 40718369, 43218285, 41357670, 41289132, 41336265, 41321697, 36055506, 43185407, 36880262, 41323011, 41410125, 41337047, 43826988, 43625771, 21104744, 43843279, 42873628, 40947256, 41277963, 43698952, 41389497, 35745722, 44177335, 2055910, 41331031, 43190428, 41184764, 41358600, 41164806, 40999368, 41030384, 41184747, 21107535, 41344172, 41367141, 41451314, 40837887, 42934833, 41197422, 40732454, 41247089, 19106649, 36892757, 19113806, 41336919, 43800055, 21035896, 35770615, 41026357, 43680134, 43175063, 41247088, 43163783, 21021128, 19034804, 41309159, 43681471, 41338118, 40825342, 41309162, 43589938, 19064110, 41307266, 40930150, 41184777, 41039425, 41040667, 43157580, 41339807, 41409722, 40836270, 41091084, 43179529, 41336463, 21032184, 41159961, 43210219, 21128912, 1592247, 41391883, 40085520, 40028577, 43627102, 41120302, 43645512, 42934720, 43582147, 36057004, 35778171, 42934746, 21169758, 40927828, 41182938, 41249174, 43601431, 41334411, 40972044, 41091063, 43177433, 21148672, 19063670, 41151891, 41358626, 43744805, 41332224, 43199412, 21035954, 43257280, 43636590, 35897675, 41359914, 41412693, 40749437, 41331041, 41334509, 41400039, 41330196, 41330935, 21096671, 41229382, 35155829, 43666707, 41053298, 41330356, 41352450, 40729782, 42483100, 43618477, 41346541, 41412811, 41335787, 40862772, 41334054, 43170480, 43034996, 41009583, 41335540, 41375307, 41118752, 21079893, 41331863, 41020686, 35606556, 40729264, 41336745, 41458914, 2030883, 41355676, 43841666, 40855736, 41358624, 41057535, 43174732, 41270546, 43763473, 41357508, 35753994, 35138037, 41363427, 40852484, 41341344, 41336937, 41330645, 36278104, 2054973, 41331752, 43600502, 41332341, 41259305, 41334318, 41343197, 36056987, 40841330, 21126521, 40993739, 43860177, 41331290, 40915849, 40716044, 41122222, 40872512, 43698066, 43188398, 41329703, 41132327, 41386465, 41333063, 35778594, 40991743, 1832263, 43637878, 35884664, 43593553, 41336072, 41332661, 41350644, 41333245, 41338298, 41028278, 43179238, 41343719, 43697931, 41330996, 40917949, 42934841, 41247129, 41342012, 40234808, 41003392, 2057293, 41336737, 43034984, 40234104, 35753491, 21040699, 920627, 43654692, 41337505, 43024861, 41269063, 41345175, 43769845, 41240348, 19135074, 21071599, 35752984, 40903513, 41344984, 41008353, 41184776, 41338121, 43582467, 41276148, 40749475, 41333356, 43034989, 41359301, 2024034, 40941062, 19076135, 41166059, 44111165, 41337549, 40839458, 44127187, 43787753, 19108508, 44044410, 43638305, 41216063, 43770792, 19067794, 41333359, 40903530, 41420715, 36781249, 41177871, 43285918, 35830396, 2042095, 41191095, 41328467, 41102120, 41240327, 43637802, 21067780, 43800678, 41320466, 44120815, 43208134, 41328002, 42708926, 41333020, 40884948, 35770222, 43196370, 43295802, 35776346, 40990362, 41209658, 41212680, 41287945, 43836347, 41338577, 41102131, 35778050, 40896815, 43734171, 43853798, 41030377, 21124133, 21094837, 41328196, 42934843, 44035925, 41021330, 41336620, 35745973, 43682217, 41445786, 41242605, 41070711, 44163553, 41313064, 43133390, 41330429, 35897681, 40895540, 40972041, 43517863, 43691574, 41065622, 41329008, 43526085, 41337901, 41331498, 43620190, 41146898, 43168167, 43618485, 41051895, 43823809, 41157583, 41237749, 41241643, 43852576, 41338629, 43698959, 35897667, 41040661, 41358914, 40866338, 36895167, 40884944, 43268145, 40997191, 41332161, 41115373, 41301634, 41413268, 36277807, 41338836, 41358329, 40867952, 35139535, 41042730, 43028403, 41339921, 35139021, 43755001, 40997217, 43168170, 21065418, 41323805, 41445548, 41408765, 41332653, 41334193, 41334173, 43700234, 43601717, 41083496, 40026667, 41336631, 41342514, 43215796, 41291287, 42953131, 43769887, 41342741, 43596191, 41028263, 35143202, 41196155, 41227222, 40234802, 44064007, 41216065, 41334690, 43817761, 43157316, 41356599, 43613381, 43637880, 41332679, 41351897, 41277942, 41351377, 41329757, 43135149, 41216106, 2055911, 40073198, 21050393, 43663392, 41337372, 41332845, 586734, 40964035, 35765903, 42934860, 41345842, 21138780, 41409507, 40085194, 21109451, 40749454, 41376076, 40997207, 43698067, 43788952, 41345174, 43173117, 42934744, 41124303, 40927825, 40874690, 41331433, 41344503, 42939680, 43166401, 19001447, 19135073, 41451319, 41249183, 41028246, 40946045, 21169766, 41339870, 43179531, 41124301, 41327435, 40028575, 43601716, 43600445, 40949297, 40949294, 41320475, 40725283, 19024534, 43267299, 41196157, 40821116, 19040156, 41309164, 40729266, 41184765, 44049597, 41198300, 36884768, 41276139, 42479181, 43654469, 41330749, 41372335, 21128900, 41331410, 40738854, 19034646, 41196148, 42948507, 41328012, 41335523, 41409001, 41334412, 41359852, 42934719, 41356701, 41332396, 43280480, 44088198, 41271858, 41336346, 35749964, 40049688, 42941644, 41356652, 40729839, 41330866, 43211977, 41358412, 41334869, 41376387, 41333979, 41239697, 44047938, 40887272, 41340272, 41271180, 41071917, 40234035, 44114045, 40997236, 36781219, 43600178, 43271749, 21048518, 41376071, 41328836, 35886751, 43842033, 35765907, 44816141, 40946038, 43799565, 36888678, 40044443, 41330127, 40731078, 35763613, 41331201, 41052850, 43191810, 19039286, 41023043, 43726447, 40841345, 41224119, 43211992, 41311332, 42715522, 41151888, 41330347, 40914662, 43179407, 41009598, 35787107, 41122189, 41335997, 41191074, 35134741, 41339232, 43208132, 44126000, 41388305, 41216069, 40822140, 40946034, 40936894, 41327742, 41238092, 41120312, 41331291, 21097176, 36888677, 19130269, 43032967, 21117812, 41324808, 41179292, 1518259, 41339297, 41385224, 40060710, 41177226, 21128925, 44101652, 21089679, 43701695, 40947268, 41330400, 41245190, 40874700, 42948550, 19039383, 41191083, 41220781, 41114700, 21119144, 40968030, 40997233, 41351325, 1832149, 43157299, 41456398, 21175218, 41309146, 41227238, 21045802, 21067387, 41345899, 43816993, 40729771, 21096663, 19130165, 41284135, 40962476, 36056998, 41340391, 40855738, 41178639, 36779831, 41458433, 41208722, 41248732, 21070274, 40876391, 41332331, 40981277, 41359986, 2042102, 41331406, 41389174, 21156527, 44083096, 41344425, 41260567, 41071927, 42921560, 35149489, 43190126, 35776069, 44198133, 35416658, 42479180, 41359698, 41360202, 40927834, 40738347, 43751850, 41338887, 43745393, 21060312, 43788571, 40234819, 21114519, 41332704, 43710174, 40974047, 41368502, 40884946, 35779599, 40729772, 35147323, 40028602, 43212002, 41359683, 19062690, 41344570, 40749534, 43735385, 21138802, 43665314, 41003390, 21091041, 41335958, 41300505, 42970583, 43682215, 41335085, 35134622, 41271202, 43855441, 43733336, 36037300, 41392577, 41426097, 41284146, 41057549, 41335264, 43600474, 43834621, 41357282, 43780415, 40934719, 36073989, 41344655, 35134073, 43782819, 40853716, 43816414, 41373092, 19135072, 41114696, 43152618, 40122600, 43697685, 41091085, 41155856, 41331659, 41302303, 35604855, 41302995, 41258028, 41133481, 40946039, 35758907, 2057309, 41153775, 41401028, 41345173, 35136857, 41334689, 35886732, 41302991, 40887052, 40717028, 35828580, 41059473, 41343583, 41091051, 19131605, 36278399, 21126828, 43834630, 42717929, 43788581, 41042784, 43177425, 21021121, 43697924, 44125583, 40997231, 21163682, 43198011, 40836818, 41334834, 19127696, 41336674, 21134011, 41327745, 41084855, 41028233, 41330129, 40835202, 41459838, 41054337, 41365296, 41337097, 36272641, 43680044, 21055574, 41133474, 40969708, 41209939, 41023663, 41084148, 41102135, 43135146, 41345703, 43680046, 40718373, 21060280, 43602204, 35161862, 40909717, 43179232, 40978243, 21119125, 41329813, 702227, 41331791, 40738838, 41286184, 21135869, 43675746, 41143996, 42934830, 21148650, 40903527, 43823821, 19074184, 41328585, 41146301, 40936900, 44164807, 41084118, 36063383, 21070183, 41034509, 41339168, 41136913, 40841350, 21061662, 41332387, 2034614, 43787722, 44195534, 43769848, 43854031, 40997249, 41008349, 42942786, 41041544, 41032119, 40950177, 41216103, 41255166, 43686370, 40716270, 43260753, 41329744, 41074061, 35767757, 41328663, 43618516, 41133488, 41196164, 40729273, 2039821, 41124302, 43672757, 40941094, 44190106, 19007062, 41330993, 41334425, 41343656, 41337093, 41028286, 43190129, 43708524, 41336054, 41332337, 43203506, 41336746, 40909709, 41353053, 40711223, 35139314, 40725279, 42934736, 43609257, 43608012, 41329748, 40869015, 43613299, 40854955, 43024866, 40833866, 41439344, 43211978, 41134789, 41335740, 43841976, 19027189, 44172486, 36781616, 41334191, 41271350, 21099594, 35780434, 43753191, 43644959, 44127624, 41209659, 40936904, 41093120, 21148654, 43823940, 41071925, 41345633, 40997177, 41412532, 40717892, 43190109, 43826708, 43146031, 43680057, 41311340, 41332848, 43611546, 41409558, 40962482, 792427, 43799561, 41402497, 793640, 43753992, 40731077, 40843475, 40841313, 41253155, 19034011, 41329059, 43279073, 40972035, 43736648, 42934857, 41216088, 35787495, 41329217, 41386053, 43589936, 41241649, 41333084, 43841990, 19067756, 21158619, 41333733, 43708483, 42720537, 41166071, 41208098, 41067723, 41128519, 41339170, 35777491, 21045810, 36257264, 41444847, 43805710, 41052537, 41067732, 41149178, 40865920, 42934716, 42722232, 42948561, 21100968, 41174949, 40883741, 40853709, 35158365, 43135150, 40965925, 41332394, 35159523, 41320478, 41009588, 43835584, 41337269, 40872530, 41412923, 41026375, 41307261, 40914661, 40841297, 41245192, 41323804, 40872527, 41209376, 41322620, 41334417, 41241037, 36063800, 41395381, 41174951, 43654337, 41150246, 35777566, 41237413, 41332325, 40989731, 41396547, 40073436, 21169760, 43601723, 40749460, 43751834, 41353055, 41327960, 41028260, 43583666, 35161377, 43769843, 35749315, 41309180, 41061508, 44131635, 41355226, 40872476, 40234823, 43717266, 40870604, 41337706, 41334897, 41330551, 43645394, 1831857, 41133484, 41159918, 41338049, 40228302, 36279364, 40855740, 41206552, 41120301, 21138803, 41059480, 40738364, 21097155, 43024870, 43644113, 40932766, 41331792, 40897396, 41335043, 19067757, 41330218, 43636592, 41330194, 41335005, 44050990, 40958456, 41354094, 21081306, 35777965, 41328416, 40821416, 43824101, 2904166, 41354608, 41346137, 41196162, 41483639, 21059467, 43518095, 2030653, 41338010, 41279999, 21100965, 41279991, 41343267, 43673759, 41342014, 40867948, 43665783, 36063362, 41371361, 41333832, 41337409, 43780406, 35851464, 36054467, 41340266, 40914665, 40717876, 44196561, 43726989, 40872516, 35604730, 43673357, 41402762, 41345198, 41336626, 40717875, 44189368, 43591935, 41333091, 41339506, 2031878, 40732459, 40018865, 1973097, 21050460, 41089083, 41226019, 41353812, 40841325, 44060549, 44064005, 43638306, 2034631, 43718634, 40898093, 40965919, 43715862, 43798368, 40900087, 43600468, 41147076, 41114705, 41230581, 40883727, 43024862, 40823097, 40931210, 21173453, 42953124, 40883728, 35770221, 41367285, 43629650, 21038514, 41418172, 42922377, 40901663, 1592183, 35775940, 41304569, 43726441, 43801291, 21128914, 41351895, 35767220, 43861614, 21148670, 40729777, 2034634, 43258877, 41357509, 43591933, 43146039, 36074598, 41177873, 43771229, 35742295, 41070702, 43625791, 41330550, 40738348, 41332442, 40716046, 41184771, 41354880, 19034806, 21163563, 42714141, 43823908, 41329746, 1832262, 40865808, 41327485, 41345674, 41358460, 19034807, 41482660, 43613582, 35774751, 41327926, 41124305, 41353231, 41230584, 43146326, 19034328, 43680968, 41245189, 21022442, 43591336, 43135452, 43805542, 21081307, 40947247, 40930142, 21079972, 21127001, 41340332, 44131414, 41222185, 21153876, 2042111, 40974041, 41372627, 41330656, 43293415, 43752693, 40718376, 40941061, 43135148, 21169773, 43269634, 43693485, 41334092, 41388099, 21061656, 41331665, 41133476, 41315395, 35886733, 41329113, 41329625, 43733685, 21119132, 41363176, 40729790, 21117268, 43816992, 43285941, 41336266, 40903533, 43153589, 40718925, 21145720, 43146030, 41332927, 21126742, 44069430, 21099577, 41149175, 43146015, 41227228, 40725709, 41184753, 41334701, 43709764, 43590130, 41277959, 43174739, 41146903, 42934727, 41338119, 41274532, 41065620, 41302993, 44114492, 35157098, 41052539, 40233165, 41385876, 43698961, 41369875, 41345844, 40934739, 41005433, 43600510, 41253142, 41036489, 41334432, 43210212, 42922391, 43853568, 43607914, 40880800, 43590806, 41341433, 40749445, 41334244, 41122216, 41337710, 36895129, 40903552, 41114701, 41332929, 21099591, 41321705, 21089691, 40876372, 2057236, 41084862, 41328039, 43680133, 41337838, 35779140, 41329657, 41392919, 41412924, 41091052, 41459518, 41340268, 41342543, 44109971, 21030831, 41354095, 44064009, 2057299, 40847597, 40854956, 35146508, 41358196, 43607904, 43177428, 40077887, 43602813, 19034645, 40907303, 40976988, 41334192, 44062054, 42934869, 41130579, 43212281, 41083494, 41082383, 41340918, 43024865, 40836817, 41354141, 41332778, 21079958, 41330195, 21027990, 40847594, 41337981, 41336485, 43159425, 40725271, 41210055, 43028397, 41328169, 41300503, 40957281, 19006964, 41057540, 35604734, 41331050, 40884957, 41353314, 41334053, 41366253, 41409141, 40934726, 41345133, 40946042, 40073185, 43210216, 43602815, 43853571, 41418401, 43721391, 41354331, 41331049, 41055984, 2054958, 43603414, 41328583, 40883737, 41356653, 42934842, 21130237, 43680137, 40874697, 41155870, 36260213, 21148663, 41408207, 21032190, 41342398, 40725278, 42708927, 40738352, 41327792, 41402275, 41228468, 41345313, 40872543, 35757660, 40863420, 43593561, 41184789, 41329528, 40995311, 41102123, 36056989, 41357638, 36895256, 43733863, 41135695, 41408042, 43823942, 41323809, 35886750, 41028272, 21175338, 19106650, 41333773, 41071929, 35158695, 43589806, 40878682, 35778819, 21028107, 36278288, 35865016, 41333938, 41049717, 43780793, 43751766, 21060296, 43690794, 41177218, 41301629, 41397401, 36063375, 41356530, 41184744, 40965909, 43817388, 41261307, 40903544, 41335270, 41336423, 40932761, 36278416, 2039823, 21158620, 40832390, 41329050, 43218547, 43029930, 41334603, 41247096, 43674230, 41321696, 21050471, 41339375, 21058268, 41331409, 41003397, 975504, 41034500, 41329065, 43164310, 41177877, 43762319, 43806977, 41227237, 43589897, 21021112, 43210211, 43655794, 41330547, 2039847, 41336349, 41097180, 41329745, 35745114, 41362222, 41160384, 43672771, 40843464, 41410222, 41331032, 41135700, 41290401, 41028283, 43819422, 43269588, 19067753, 43834632, 41309138, 44113507, 44191590, 41344228, 41341918, 43701234, 43798726, 43211993]
    med = "steroids"
    sql = f"concept_id in ({','.join(map(str,steroid_concept_ids))})"
    print("  steroids sql = ",sql)
    dm_t = concept.filter(sql)
    df_steroids = dm_t.selectExpr(f"'{med}' as codeset_id", "concept_id", f"'{med}' as concept_set_name", "cast(1 as boolean) as is_most_recent_version", "1 as version", "concept_name", "cast(1 as boolean) as archived")
    print("Adding concept_ids for ",med,df_steroids.count())
    df_steroids.show()
    
    # Add existing N3C codesets
    df = concept_set_members

    df = df.union(ckd45).union(height_weight_bmi).union(df_steroids).union(comparator_meds_all)
    df = df.union(all_cond)

    df = df.dropDuplicates()
    
    return df
    
    
    

