


-- 查找所有与肾衰竭（kidney failure,renal failure）相关的主要疾病
CREATE TABLE IF NOT EXISTS public.kidneyfailure_maindiseases AS
SELECT
	* 
FROM
	mimic_hosp.d_icd_diagnoses 
WHERE
	( long_title LIKE'%kidney failure%' 
	OR long_title LIKE'%renal failure%' 
	OR long_title LIKE'%Kidney failure%' 
	OR long_title LIKE'%Renal failure%' ) 
	AND long_title NOT LIKE'%complicated by renal failure%';
	
	
-- 查询患这些主要疾病（肾功能衰竭）的患者
CREATE TABLE IF	NOT EXISTS kidneyfailure_patients AS 
SELECT
	d1.*,
	k1.long_title 
FROM
	mimic_hosp.diagnoses_icd d1
	LEFT JOIN PUBLIC.kidneyfailure_maindiseases k1 ON d1.icd_code = k1.icd_code 
WHERE
	d1.icd_code IN ( SELECT icd_code FROM PUBLIC.kidneyfailure_maindiseases );

select count(distinct subject_id) from kidneyfailure_patients;




-- 计算病症的病例统计结果
CREATE TABLE IF NOT EXISTS PUBLIC.kidneyfailure_diseafreq AS
select k1.icd_code,k1.long_title,count(k1.subject_id) as count_diseases
from public.kidneyfailure_patients k1
GROUP BY k1.icd_code,k1.long_title
ORDER BY count_diseases desc;


-- 计算病症的优先级统计结果
CREATE TABLE IF NOT EXISTS PUBLIC.kidneyfailure_diseaseseqnum AS
SELECT
	k1.seq_num,
	COUNT ( k1.subject_id ) AS count_seq 
FROM
	PUBLIC.kidneyfailure_patients k1 
GROUP BY
	k1.seq_num 
ORDER BY
	count_seq DESC;


-- 筛选肾功能衰竭为前3主要病症的患者【37613】
CREATE TABLE IF NOT EXISTS PUBLIC.kidneyfailure_respatients AS 
SELECT
	k1.subject_id,
	k1.hadm_id,
	MIN ( k1.seq_num ) AS seq_num,
	MAX ( k1.icd_code ) AS icd_code,
	MAX ( k1.icd_version ) AS icd_version,
	MAX ( k1.long_title ) AS long_title 
FROM
	PUBLIC.kidneyfailure_patients k1 
WHERE
	k1.seq_num <= 3 
GROUP BY
	k1.subject_id,
	k1.hadm_id;

select count(distinct subject_id) from kidneyfailure_respatients;


-- 患者的基本入院信息【37613】
CREATE TABLE IF NOT EXISTS PUBLIC.info_dm_admis AS 
SELECT
	k1.*,
	a2.age,
	p1.gender,
	a1.marital_status,
	a1.ethnicity,
	
	a1.admittime,
	a1.dischtime,
	a1.edregtime,
	a1.edouttime,
	a1.deathtime,
	(datetime_diff(a1.dischtime,a1.admittime,'HOUR')/24.0) as admitcustime,
	case when a1.edregtime is not null then 
			(datetime_diff(a1.edregtime,a1.edouttime,'HOUR')/24.0) else 0 end as edcustime,
	a1.hospital_expire_flag as death_flag,
	case when a1.edregtime is not null then 1 else 0 end as ed_flag,
	case when i1.subject_id is not null then 1 else 0 end as icu_flag

FROM
	PUBLIC.kidneyfailure_respatients k1
	LEFT JOIN mimic_core.admissions a1 ON k1.subject_id = a1.subject_id AND k1.hadm_id = a1.hadm_id
	LEFT JOIN PUBLIC.age a2 ON a1.subject_id = a2.subject_id AND a1.hadm_id = a2.hadm_id
	LEFT JOIN mimic_core.patients p1 ON p1.subject_id = k1.subject_id
	LEFT JOIN ( SELECT icustays.subject_id, icustays.hadm_id 
							FROM mimic_icu.icustays 
							GROUP BY icustays.subject_id, icustays.hadm_id ) i1 ON i1.subject_id = k1.subject_id 
AND i1.hadm_id = k1.hadm_id 
ORDER BY
	k1.subject_id,
	k1.hadm_id,
	a1.admittime;



-- 筛选进入了ICU的患者并查询ICU相关信息【9344】
CREATE TABLE IF NOT EXISTS PUBLIC.info_icu_stay AS 
SELECT
		i1.subject_id,
		i1.hadm_id,
		COUNT ( i1.stay_id ) AS icu_stay_count,
		SUM ( i1.los ) AS icu_stay_custime,
		MAX ( i1.first_careunit ) AS first_careunit,
		MAX ( i1.last_careunit ) AS last_careunit,
			-- 外科重症监护室(SICU)
			-- 心脏血管重症监护室(CVICU)
			-- 医疗重症监护室(MICU)
			-- 内科/外科重症监护室(MICU/SICU)
		MAX ( CASE WHEN first_careunit = 'Coronary Care Unit (CCU)' 
									OR last_careunit = 'Coronary Care Unit (CCU)' 
					THEN 1 ELSE 0 END ) AS flag_CCU,
		MAX ( CASE WHEN first_careunit = 'Surgical Intensive Care Unit (SICU)' 
									OR last_careunit = 'Surgical Intensive Care Unit (SICU)' 
					THEN 1 ELSE 0 END ) AS flag_SICU,
		MAX ( CASE WHEN first_careunit = 'Neuro Intermediate' 
									OR last_careunit = 'Neuro Intermediate' 
					THEN 1 ELSE 0 END ) AS flag_NI,
		MAX ( CASE WHEN first_careunit = 'Neuro Stepdown' 
									OR last_careunit = 'Neuro Stepdown' 
					THEN 1 ELSE 0 END ) AS flag_NS,
		MAX ( CASE WHEN first_careunit = 'Neuro Surgical Intensive Care Unit (Neuro SICU)' 
									OR last_careunit = 'Neuro Surgical Intensive Care Unit (Neuro SICU)' 
					THEN 1 ELSE 0 END ) AS flag_NSICU,
		MAX ( CASE WHEN first_careunit = 'Cardiac Vascular Intensive Care Unit (CVICU)' 
									OR last_careunit = 'Cardiac Vascular Intensive Care Unit (CVICU)' 
					THEN 1 ELSE 0 END ) AS flag_CVIU,
		MAX ( CASE WHEN first_careunit = 'Trauma SICU (TSICU)'
									OR last_careunit = 'Trauma SICU (TSICU)' 
					THEN 1 ELSE 0 END ) AS flag_TSICU,
		MAX ( CASE WHEN first_careunit = 'Medical Intensive Care Unit (MICU)' 
									OR last_careunit = 'Medical Intensive Care Unit (MICU)' 
					THEN 1 ELSE 0 END ) AS flag_MICU,
		MAX ( CASE WHEN first_careunit = 'Medical/Surgical Intensive Care Unit (MICU/SICU)'
									OR last_careunit = 'Medical/Surgical Intensive Care Unit (MICU/SICU)' 
					THEN 1 ELSE 0 END ) AS flag_MSICU 
FROM
	mimic_icu.icustays i1
WHERE
	i1.hadm_id IN ( SELECT hadm_id FROM info_dm_admis ) 
	AND i1.subject_id IN ( SELECT subject_id FROM info_dm_admis ) 
GROUP BY
	i1.subject_id,
	i1.hadm_id 
ORDER BY
	i1.subject_id,
	i1.hadm_id;


-- 所有的icu事件，已筛选患者中
CREATE TABLE IF NOT EXISTS PUBLIC.info_icu_allpati AS 
SELECT
		i1.subject_id,
		i1.hadm_id,
		i1.stay_id
FROM
	mimic_icu.icustays i1
WHERE
	i1.hadm_id IN ( SELECT hadm_id FROM info_dm_admis ) 
	AND i1.subject_id IN ( SELECT subject_id FROM info_dm_admis ) 
GROUP BY
	i1.subject_id,
	i1.hadm_id,
	i1.stay_id
ORDER BY
	i1.subject_id,
	i1.hadm_id,
	i1.stay_id;




-- 用药信息特征

-- 抗生素【37613】
CREATE TABLE IF NOT EXISTS PUBLIC.info_adm_antibiotic AS 
select i1.subject_id,i1.hadm_id,
			case when t1.subject_id is not null then 1 else 0 end as antib_flag,
			t1.antib_route
from public.info_dm_admis i1
	left join (select a1.subject_id,a1.hadm_id,max(a1.route) as antib_route
						from public.antibiotic a1
						GROUP BY	a1.subject_id,a1.hadm_id) t1 
	on i1.subject_id=t1.subject_id and i1.hadm_id=t1.hadm_id;




-- 其他药物inputflag，ICU中数据【10514】
CREATE TABLE IF NOT EXISTS PUBLIC.info_icu_druginput AS 
select i1.subject_id,i1.hadm_id,i1.stay_id,
	case when m1.stay_id is not null then 1 else 0 end as dobutamine_flag,
	case when m2.stay_id is not null then 1 else 0 end as dopamine_flag,
	case when m3.stay_id is not null then 1 else 0 end as neuroblock_flag,
	case when m4.stay_id is not null then 1 else 0 end as epinephrine_flag,
	case when m5.stay_id is not null then 1 else 0 end as norepinephrine_flag,
	case when m6.stay_id is not null then 1 else 0 end as phenylephrine_flag,
	case when m7.stay_id is not null then 1 else 0 end as vasopressin_flag
from public.icustay_detail i1
		left join (select d1.stay_id from dobutamine d1 GROUP BY d1.stay_id) m1 on i1.stay_id=m1.stay_id
		left join (select d1.stay_id from dopamine d1 GROUP BY d1.stay_id) m2 on i1.stay_id=m2.stay_id
		left join (select d1.stay_id from neuroblock d1 GROUP BY d1.stay_id) m3 on i1.stay_id=m3.stay_id
		left join (select d1.stay_id from epinephrine d1 GROUP BY d1.stay_id) m4 on i1.stay_id=m4.stay_id
		left join (select d1.stay_id from norepinephrine d1 GROUP BY d1.stay_id) m5 on i1.stay_id=m5.stay_id
		left join (select d1.stay_id from phenylephrine d1 GROUP BY d1.stay_id) m6 on i1.stay_id=m6.stay_id
		left join (select d1.stay_id from vasopressin d1 GROUP BY d1.stay_id) m7 on i1.stay_id=m7.stay_id
where i1.subject_id in (select subject_id from public.info_icu_stay) and
			i1.hadm_id in (select hadm_id from public.info_icu_stay);

-- select count(distinct subject_id) from info_icu_druginput;


-- 实验室检查项目信息特征【10514】
CREATE TABLE IF NOT EXISTS PUBLIC.info_icu_laball AS 
SELECT i1.hadm_id,
	l2.*,
	l1.lactate_min,
	l1.lactate_max,
	l1.ph_min,
	l1.ph_max,
	l1.so2_min,
	l1.so2_max,
	l1.po2_min,
	l1.po2_max,
	l1.pco2_min,
	l1.pco2_max,
	l1.aado2_min,
	l1.aado2_max,
	l1.aado2_calc_min,
	l1.aado2_calc_max,
	l1.pao2fio2ratio_min,
	l1.pao2fio2ratio_max,
	l1.baseexcess_min,
	l1.baseexcess_max,
	l1.totalco2_min,
	l1.totalco2_max,
	l3.dialysis_present,
	l3.dialysis_active,
	l3.dialysis_type,
	l4.urineoutput 
FROM
	PUBLIC.info_icu_allpati i1
	LEFT JOIN first_day_bg l1 ON i1.stay_id = l1.stay_id -- 气血分析： Highest/lowest blood gas values for arterial blood specimens
	INNER JOIN first_day_lab l2 ON i1.stay_id = l2.stay_id -- 实验室检查项目
	INNER JOIN first_day_rrt l3 ON i1.stay_id = l3.stay_id -- 透析：flag indicating if patients received dialysis during
	INNER JOIN first_day_urine_output l4 ON i1.stay_id = l4.stay_id;--总尿量：Total urine output


-- 一般检查信息特征
CREATE TABLE IF NOT EXISTS PUBLIC.info_icu_vitalsign AS 
SELECT
	i1.hadm_id,
	b2.*,
	b1.height,
	b3.weight 
FROM
	info_icu_allpati i1
	LEFT JOIN first_day_height b1 ON i1.stay_id = b1.stay_id -- 身高
	INNER JOIN first_day_vitalsign b2 ON i1.stay_id = b2.stay_id -- 生命体征：vital signs
	INNER JOIN first_day_weight b3 ON i1.stay_id = b3.stay_id;--体重



-- 常用重症评分【10514】
CREATE TABLE IF NOT EXISTS PUBLIC.info_icu_allscore AS 
SELECT
	s2.*,
	i1.hadm_id,
	s1.sofa,
	s3.apsiii,
	s4.lods,
	s5.oasis,
	s6.sapsii,
	s7.sirs 
FROM
	info_icu_allpati i1
	INNER JOIN first_day_sofa s1 ON i1.stay_id = s1.stay_id --SOFA：序贯器官衰竭评估(SOFA)
	INNER JOIN first_day_gcs s2 ON i1.stay_id = s2.stay_id -- Glasgow Coma Scale, a measure of neurological function.
	INNER JOIN apsiii s3 ON i1.stay_id = s3.stay_id --急性生理评分III (APS III)
	INNER JOIN lods s4 ON i1.stay_id = s4.stay_id --Logistic器官功能障碍评分(LODS)
	INNER JOIN oasis s5 ON i1.stay_id = s5.stay_id --牛津急性疾病严重程度评分(OASIS)
	INNER JOIN sapsii s6 ON i1.stay_id = s6.stay_id --简化急性生理评分II (SAPS II)
	INNER JOIN sirs s7 ON i1.stay_id = s7.stay_id;--全身炎症反应综合征(SIRS)标准



-- 并发症【37613】
CREATE TABLE IF NOT EXISTS PUBLIC.info_adm_complication AS 
SELECT
	c1.*
FROM
	info_dm_admis i1
	INNER JOIN charlson c1 ON i1.subject_id = c1.subject_id 
	AND i1.hadm_id = c1.hadm_id;



