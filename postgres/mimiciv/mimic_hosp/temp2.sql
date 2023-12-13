SELECT
	mimic_icu.d_items.*
FROM
	mimic_icu.d_items
WHERE
	mimic_icu.d_items.itemid IN (220045,225309,225310,225312,220050,220051,220052,220179,220180,220181,220210,224690,220277,225664,220621,226537,223762,223761,224642);
	
	
select *
from mimic_icu.d_items
where mimic_icu.d_items.category in
('Dialysis','Labs','Routine Vital Signs','Antibiotics',
'Medications','Dialysis')
ORDER BY mimic_icu.d_items.category;





