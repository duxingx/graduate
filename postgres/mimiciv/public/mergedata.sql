CREATE TABLE IF NOT EXISTS PUBLIC.res_adm AS 
SELECT
	iac.*, 
	iaa.antib_flag, 	iaa.antib_route, 
	ida.seq_num, 	ida.icd_code, 	ida.icd_version, 	ida.long_title, 
	ida.age, 	ida.gender, 	ida.marital_status, 
	ida.ethnicity, 	ida.admittime, 	ida.dischtime, 
	ida.edregtime, 	ida.edouttime, 	ida.deathtime, 	ida.admitcustime, 
	ida.edcustime, 	ida.death_flag, 	ida.icu_flag, 	ida.ed_flag 
FROM
	info_dm_admis ida
	INNER JOIN info_adm_antibiotic iaa ON ida.subject_id = iaa.subject_id 
	AND ida.hadm_id = iaa.hadm_id
	INNER JOIN info_adm_complication iac ON ida.subject_id = iac.subject_id 
	AND ida.hadm_id = iac.hadm_id;




CREATE TABLE IF NOT EXISTS PUBLIC.res_icu1 AS 
SELECT
	iil.*, 
	iiv.heart_rate_min, 	iiv.heart_rate_max, 	iiv.heart_rate_mean, 	iiv.sbp_min, 	iiv.sbp_max, 
	iiv.sbp_mean, 	iiv.dbp_min, 	iiv.dbp_max, 	iiv.dbp_mean, 	iiv.mbp_min, 	iiv.mbp_max, 
	iiv.mbp_mean, 	iiv.resp_rate_min, 	iiv.resp_rate_max,	iiv.resp_rate_mean, 	iiv.temperature_min, 
	iiv.temperature_max, 	iiv.temperature_mean, 	iiv.spo2_min, 	iiv.spo2_max, 
	iiv.spo2_mean, iiv.height, 	iiv.weight, 
	iid.dobutamine_flag, 	iid.dopamine_flag, 	iid.neuroblock_flag, 	iid.epinephrine_flag, 
	iid.norepinephrine_flag, 	iid.phenylephrine_flag, 	iid.vasopressin_flag, 
	iia.gcs_min, 	iia.gcs_motor, 	iia.gcs_verbal, 	iia.gcs_eyes, 	iia.gcs_unable, 
	iia.sofa, 	iia.apsiii, 	iia.lods, 	iia.oasis, 	iia.sapsii,	iia.sirs
FROM
	info_icu_allpati icu
	INNER JOIN info_icu_allscore iia ON icu.stay_id = iia.stay_id
	INNER JOIN info_icu_druginput iid ON icu.stay_id = iid.stay_id
	INNER JOIN info_icu_laball iil ON icu.stay_id = iil.stay_id
	INNER JOIN info_icu_vitalsign iiv ON icu.stay_id = iiv.stay_id;
	

CREATE TABLE IF NOT EXISTS PUBLIC.res_icu2 AS 
SELECT
	ri.*,
	iis.icu_stay_count,
	iis.icu_stay_custime,
	iis.first_careunit,
	iis.last_careunit,
	iis.flag_ccu,
	iis.flag_sicu,
	iis.flag_ni,
	iis.flag_ns,
	iis.flag_nsicu,
	iis.flag_cviu,
	iis.flag_tsicu,
	iis.flag_micu,
	iis.flag_msicu 
FROM
	res_icu1 ri
	LEFT JOIN info_icu_stay iis ON ri.subject_id = iis.subject_id 
	AND ri.hadm_id = iis.hadm_id;


CREATE TABLE IF NOT EXISTS PUBLIC.res_all AS 
SELECT
	ri.*,
	ra.age_score,
	ra.myocardial_infarct,	ra.congestive_heart_failure,
	ra.peripheral_vascular_disease,
	ra.cerebrovascular_disease,	ra.dementia,	ra.chronic_pulmonary_disease,
	ra.rheumatic_disease,	ra.peptic_ulcer_disease,	ra.mild_liver_disease,
	ra.diabetes_without_cc,	ra.diabetes_with_cc,
	ra.paraplegia,	ra.renal_disease,	ra.malignant_cancer,	ra.severe_liver_disease,
	ra.metastatic_solid_tumor,	ra.aids,	ra.charlson_comorbidity_index,
	ra.antib_flag,	ra.antib_route,
	ra.seq_num,	ra.icd_code,	ra.icd_version,	ra.long_title,
	ra.age,	ra.gender,	ra.marital_status,	ra.ethnicity,
	ra.admittime,	ra.dischtime,
	ra.edregtime,	ra.edouttime,	ra.deathtime,
	ra.admitcustime,	ra.edcustime,
	ra.death_flag,	ra.icu_flag,	ra.ed_flag
	
FROM
	res_icu2 ri
	LEFT JOIN res_adm ra ON ra.subject_id = ri.subject_id 
	AND ra.hadm_id = ri.hadm_id;
	