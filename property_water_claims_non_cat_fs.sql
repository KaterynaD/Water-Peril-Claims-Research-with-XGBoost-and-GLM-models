drop table if exists kdlab.property_water_claims_non_cat_fs_v5;														
create table kdlab.property_water_claims_non_cat_fs_v5 as														
with wc_PROPERTY_MODELDATA as 														
(														
select   modeldata_id														
       , systemidstart														
       , systemidend														
       , cast(date_part(year, startdate) as int) cal_year														
       , startdate														
       , enddate														
       , startdatetm														
       , enddatetm														
       , DateDiff(d, startdate, enddate )/365.25  ecy														
/*------------------------------------------POLICY DATA---------------------------------*/	   													
       , pol_policynumber PolicyNumber														
       , f.policy_uniqueid 														
       , p.POL_POLICYNUMBERSUFFIX PolicyTerm														
       , case when  p.POL_POLICYNUMBERSUFFIX='01' then 'New' else 'Renewal' end PolicyType														
       , p.pol_effectivedate effectivedate														
       , p.pol_expirationdate expirationdate														
       , p.POL_MASTERSTATE PolicyState		   												
       , pe.PolicyFormCode PolicyForm														
       , case when pe.persistencydiscountdt>'1900-01-01' then cast( datediff(month,pe.persistencydiscountdt, p.pol_effectivedate)/12 as int)else 0 end  Persistency														
       , co.comp_number CompanyCd														
       , co.comp_name1 CarrierCd														
/*---------------------------------------END POLICY DATA---------------------------------*/	 													
/*-------------------------------------- Agency  DATA   --------------------------------*/														
	   , replace(a.agency_group	, ',','') agency_group												
       , replace(a.prdr_name1 , ',','')   producername														
       , a.territory														
/*-------------------------------------- END Agency  DATA   --------------------------------*/		   												
       , risknumber														
       , risktype														
/*------------------------------------------BUILDING DATA---------------------------------*/	   													
       , YearBuilt 														
       , SqFt 														
       , Stories 														
       , upper(case when b.roofcd='~' then 'OTHER' else b.roofcd end) roofcd														
       , case when risktype='Homeowners' then NumberOfFamilies else OwnerOccupiedUnits + TenantOccupiedUnits end Units_tmp 														
	   , case when units_tmp=0 then 1 else units_tmp end units													
       , upper(case when b.occupancycd='~' then 'No' when b.occupancycd like 'Occupied%' then 'OccupiedNow' else b.occupancycd end) occupancycd														
       , case when AllperilDed 	in ('~', 'None') then 0 else cast(b.AllperilDed as int) end AllperilDed 													
       , case when b.waterded in ('~', 'None') then 0 else cast(b.waterded as int) end waterded														
       , case when b.protectionclass='~' then 0 else cast(b.protectionclass as int) end protectionclass														
	   , case 													
          when b. ConstructionCd  in ('F','AF','M','B') then   ConstructionCd 														
          else 'OTHER'														
         end  ConstructionCd 														
       , case														
         when b.reportedfirehazardscore='Extreme' then 6														
         when b.reportedfirehazardscore='High' then 4														
         when b.reportedfirehazardscore='Moderate' then 2														
         when b.reportedfirehazardscore='Low' then 1														
         when b.reportedfirehazardscore='Negligible' then 0														
         when b.reportedfirehazardscore in ('Indeterminate','~') then 														
          case														
           when b.firehazardscore='Extreme' then 6														
           when b.firehazardscore='High' then 4														
           when b.firehazardscore='Moderate' then 2														
           when b.firehazardscore='Low' then 1														
           when b.firehazardscore='Negligible' then 0														
         else -1														
        end														
       else cast(public.removenotnumeric(reportedfirehazardscore) as int)														
       end Fire_Risk_Model_Score														
       , upper(replace(pc.MultiPolicyAutoDiscount,'~','No'))  MultiPolicyInd_tmp														
	   , case when  MultiPolicyInd_tmp='NO' then 0 else 1 end MultiPolicyInd	   												
       , upper(replace(pc.MultiPolicyUmbrellaDiscount,'~','No'))  MultiPolicyIndUmbrella_tmp														
	   , case when  MultiPolicyIndUmbrella_tmp='NO' then 0 else 1 end MultiPolicyIndUmbrella	   												
       , upper(replace(earthquakeumbrellaind,'~','No'))  earthquakeumbrellaind_tmp														
	   , case when  earthquakeumbrellaind_tmp='NO' then 0 else 1 end earthquakeumbrellaind	   												
	   , case when upper(CovADDRR_SecondaryResidence) in ('NO','~') then 0 else 1 end secondaryresidence													
	   , upper(case when secondaryresidence=1 then 'Secondary' else UsageType  end) 	UsageType												
       , OrdinanceOrLawPct 														
       , upper(replace(FunctionalReplacementCost,'~','No'))  FunctionalReplacementCost_tmp														
	   , case when FunctionalReplacementCost_tmp='NO' then 0 else 1 end FunctionalReplacementCost													
       , upper(replace(homegardcreditind,'~','No'))  homegardcreditind_tmp														
	   , case when  homegardcreditind_tmp='NO' then 0 else 1 end homegardcreditind													
       , upper(replace(sprinklersystem,'~','No'))  sprinklersystem_tmp														
       , case when case when sprinklersystem_tmp='NONE' then 'NO' else sprinklersystem_tmp end='NO' then 0 else 1 end sprinklersystem														
       , upper(replace(pc.LandlordInd,'~','No'))  landlordind_tmp														
	   , case when landlordind_tmp='NO' then 0 else 1 end landlordind													
       , upper(replace(rentersinsurance,'~','No'))  RentersInsurance_tmp														
	   , case when RentersInsurance_tmp='NO' then 0 else 1 end RentersInsurance													
       , upper(case when FireAlarmType='~' then 'No' when upper(FireAlarmType)='NONE' then 'No' else 'Yes' end)  FireAlarmType_tmp														
	   , case when FireAlarmType_tmp='NO' then 0 else 1 end FireAlarmType													
	   , upper(case when BurglaryAlarmType='~' then 'No' when upper(BurglaryAlarmType)='NONE' then 'No' else 'Yes' end)  BurglaryAlarmType_tmp													
	   , case when BurglaryAlarmType_tmp='NO' then 0 else 1 end BurglaryAlarmType	   												
       , upper(replace(WaterDetectionDevice,'~','No'))  WaterDetectionDevice_tmp														
	   , case when WaterDetectionDevice_tmp='NO' then 0 else 1 end WaterDetectionDevice		   											
       , upper(replace(NeighborhoodCrimeWatchInd,'~','No'))  NeighborhoodCrimeWatchInd_tmp														
	   , case when NeighborhoodCrimeWatchInd_tmp='NO' then 0 else 1 end NeighborhoodCrimeWatchInd		   											
       , upper(replace(PropertyManager,'~','No'))  PropertyManager_tmp														
	   , case when PropertyManager_tmp='NO' then 0 else 1 end PropertyManager	   												
       , upper(replace(SafeguardPlusInd,'~','No'))  SafeguardPlusInd_tmp														
	   , case when SafeguardPlusInd_tmp='NO' then 0 else 1 end SafeguardPlusInd	   												
	   , upper(replace(kitchenfireextinguisherind,'~','No')) kitchenfireextinguisherind_tmp													
	   , case when kitchenfireextinguisherind_tmp='NO' then 0 else 1 end kitchenfireextinguisherind	 	   											
	   , upper(replace(gatedcommunityind,'~','No')) gatedcommunityind_tmp													
	   , case when gatedcommunityind_tmp='NO' then 0 else 1 end gatedcommunityind	   												
	   , upper(replace(deadboltind,'~','No')) deadboltind_tmp													
	   , case when deadboltind_tmp='NO' then 0 else 1 end deadboltind	   												
	   , upper(replace(poolind,'~','No')) poolind_tmp													
	   , case when poolind_tmp='NO' then 0 else 1 end poolind	   												
	   , upper(replace(ReplacementCostDwellingInd,'~','No')) ReplacementCostDwellingInd_tmp													
	   , case when ReplacementCostDwellingInd_tmp='NO' then 0 else 1 end ReplacementCostDwellingInd													
	   , upper(replace(replacementvalueind,'~','No')) replacementvalueind_tmp													
	   , case when replacementvalueind_tmp='NO' then 0 else 1 end replacementvalueind	   												
	   , upper(replace(serviceline,'~','No')) serviceline_tmp													
	   , case when serviceline_tmp='NO' then 0 else 1 end serviceline	   												
	   , upper(replace(equipmentbreakdown,'~','No')) equipmentbreakdown_tmp													
	   , case when equipmentbreakdown_tmp='NO' then 0 else 1 end equipmentbreakdown	   												
	   , numberoffamilies	   												
/*------------------------------------------END BUILDING DATA---------------------------------*/														
/*----------------------------------          INSURED DATA      ------------------------------*/														
       , case when i.dob = '1900-01-01' then null else DateDiff(y,  i.dob,p.pol_EffectiveDate)  end InsuredAge														
	   , i.maritalstatus													
       , i.InsuranceScore														
       , i.OverriddenInsuranceScore														
       , i.InsuranceScoreValue														
       , i.insscoretiervalueband														
       , i.financialstabilitytier														
/*----------------------------------      END     INSURED DATA      ------------------------------*/       														
       , AllCov_WP														
       , CovA_WP														
	   , case when pe.renewaltermcd='1 Year' then 1 else 2 end * CovA_wp*(DateDiff(d, startdate, case when enddate > dateadd(month, -3, current_date) then dateadd(month, -3, current_date) else  enddate end)/365.25) CovA_ep													
       , public.removenotnumeric(cova_deductible) cova_deductible														
       , cova_limit														
       , cova_ic_nc_water														
       , cova_il_nc_water	   													
/*----------------------------------------Water Scores-----------------------------------------------*/	   													
       , water_risk_3_blk														
       , water_risk_fre_3_blk	   													
	   , appl_fail_3_blk													
	   , fixture_leak_3_blk													
	   , pipe_froze_3_blk													
	   , plumb_leak_3_blk													
	   , rep_cost_3_blk													
	   , ustructure_fail_3_blk													
	   , waterh_fail_3_blk													
	   , water_risk_sev_3_blk													
/*-------------------------------------------------------------------------------------------------*/	   	   												
       , f.loaddate														
/*-------------------------------------------------------------------------------------------------*/	   													
       , isnull(cnt_ap.Cnt_Active_Other_Policies,1)	customer_cnt_active_policies   													
from FSBI_DW_SPINN.FACT_HO_LL_MODELDATA f														
join FSBI_DW_SPINN.DIM_POLICY p														
on f.policy_id=p.policy_id														
join FSBI_DW_SPINN.DIM_BUILDING b														
on  f.policy_id=b.policy_id														
and f.building_id=b.building_id														
join FSBI_DW_SPINN.VDIM_PRODUCER a														
on f.producer_id=a.producer_id														
join FSBI_DW_SPINN.DIM_POLICYEXTENSION pe														
on f.policy_id=pe.policy_id														
join FSBI_DW_SPINN.DIM_INSURED i														
on f.policy_id=i.policy_id														
join FSBI_DW_SPINN.VDIM_COMPANY co														
on p.company_id=co.company_id														
join FSBI_DW_SPINN.dim_policy_changes pc														
on f.policy_changes_id=pc.policy_changes_id														
and f.policy_id=pc.policy_id														
left outer join fsbi_stg_spinn.tmp_customer_cnt_active_policies cnt_ap														
on cnt_ap.policy_id=f.policy_id														
where cal_year>2008														
and yearbuilt >=1850 and yearbuilt<=2020														
and sqft>=600 and sqft<=10000														
and cova_wp>0														
and ecy>0														
and policystate<>'UT'														
)														
select	distinct 													
         modeldata_id														
       , systemidstart														
       , systemidend														
       , cal_year														
       , startdate														
       , enddate														
       , startdatetm														
       , enddatetm														
       , ecy														
	   , ln(cast(ecy as float)) log_ecy													
/*------------------------------------------POLICY DATA---------------------------------*/	   													
       , PolicyNumber														
       , policy_uniqueid 														
       , PolicyTerm														
       , PolicyType														
       , effectivedate														
       , expirationdate														
       , PolicyState		   												
       , PolicyForm														
       , Persistency														
       , CompanyCd														
       , CarrierCd														
/*---------------------------------------END POLICY DATA---------------------------------*/	 													
/*-------------------------------------- Agency  DATA   --------------------------------*/														
	   , agency_group													
       , producername														
       , territory														
/*-------------------------------------- END Agency  DATA   --------------------------------*/		   												
       , risknumber														
       , risktype														
/*------------------------------------------BUILDING DATA---------------------------------*/	   													
       ,  case														
		      when yearbuilt<=1900 then 1900												
		  when yearbuilt>1900 and yearbuilt<=1905 then 1905												
		          when yearbuilt>1905 and yearbuilt<=1910 then 1910												
		  when yearbuilt>1910 and yearbuilt<=1915 then 1915												
		  when yearbuilt>1915 and yearbuilt<=1920 then 1920												
		  when yearbuilt>1920 and yearbuilt<=1925 then 1925												
		  when yearbuilt>1925 and yearbuilt<=1930 then 1930												
		  when yearbuilt>1930 and yearbuilt<=1935 then 1935												
		  when yearbuilt>1935 and yearbuilt<=1940 then 1940												
		  when yearbuilt>1940 and yearbuilt<=1945 then 1945												
		  when yearbuilt>1945 and yearbuilt<=2007 then yearbuilt												
		  when yearbuilt>2007 and  yearbuilt<=2009 then 2009		  										
		  when yearbuilt>2009 and  yearbuilt<=2015 then 2015												
		  else 2019												
	     end yearbuilt													
	   ,ln(cast(yearbuilt as float)) log_yearbuilt	 												
       ,case 														
	    when ceil(sqft / 100) * 100<=800 then 800													
	    when ceil(sqft / 100) * 100>800 and ceil(sqft / 100) * 100<=3200 then ceil(sqft / 100) * 100													
	    when ceil(sqft / 100) * 100>3200 and ceil(sqft / 100) * 100<=3400 then 3400													
	    when ceil(sqft / 100) * 100>3400 and ceil(sqft / 100) * 100<=3600 then 3600													
	when ceil(sqft / 100) * 100>3600 and ceil(sqft / 100) * 100<=4000 then 4000													
	    when ceil(sqft / 100) * 100>4000 then 5000													
	   end sqft													
       , ln(cast(sqft as float)) log_sqft	   													
       , case when Stories=0 then 1 when Stories>3 then 1 else Stories end Stories														
       , roofcd														
       ,	case 													
		when roofcd='COMPO'  then 	8											
		when roofcd='TILE'  then 	7											
		when roofcd='OTHER'  then 	6											
		when roofcd='TAR'  then 	5											
		when roofcd='ASPHALT'  then 	4											
		when roofcd='MEMBRANE'  then 	3											
		when roofcd='WOOD'  then 	2											
		when roofcd='METAL'  then 	1											
        end roofcd_encd					   									
       , units														
       , occupancycd														
	   , case 													
		when occupancycd='OCCUPIEDNOW' then 1												
		when occupancycd='TENANT' then 2												
		else 3 												
         end occupancy_encd						   								
       , AllperilDed 														
       , waterded														
       , protectionclass														
	   , ConstructionCd 													
	   , case													
		when constructioncd='F' then 5												
		when constructioncd='AF' then 4												
		when constructioncd='B' then 3												
		when constructioncd='OTHER' then 2												
		when constructioncd='M' then 1												
         end constructioncd_encd						   								
       , Fire_Risk_Model_Score														
       , MultiPolicyInd	   													
	   , MultiPolicyIndUmbrella	   												
	   , earthquakeumbrellaind	   												
	   , UsageType													
	   ,	case												
		when usagetype='PRIMARY' then 7												
		when usagetype='RENTAL' then 6												
		when usagetype='COC' then 5												
		when usagetype='VACANT' then 4												
		when usagetype='SEASONAL' then 3												
		when usagetype='SECONDARY' then 2												
		when usagetype='UNOCCUPIED' then 1												
	    end usagetype_encd													
       , OrdinanceOrLawPct 														
	   , FunctionalReplacementCost													
       , homegardcreditind														
       , sprinklersystem														
       , landlordind														
       , RentersInsurance														
       , FireAlarmType														
       , BurglaryAlarmType	   													
       , WaterDetectionDevice		   												
       , NeighborhoodCrimeWatchInd		   												
       , PropertyManager	   													
       , SafeguardPlusInd	   													
	   , kitchenfireextinguisherind	 	   											
	   , gatedcommunityind	   												
	   , deadboltind	   												
	   , poolind	   												
	   , ReplacementCostDwellingInd													
	   , replacementvalueind	   												
	   , serviceline	   												
	   , equipmentbreakdown	   												
	   , numberoffamilies	   												
/*------------------------------------------END BUILDING DATA---------------------------------*/														
/*----------------------------------          INSURED DATA      ------------------------------*/														
       , InsuredAge														
	   , maritalstatus													
       , InsuranceScore														
       , OverriddenInsuranceScore														
       , InsuranceScoreValue														
       , insscoretiervalueband														
       , financialstabilitytier														
/*----------------------------------      END     INSURED DATA      ------------------------------*/       														
       , AllCov_WP														
       , CovA_WP														
	   , CovA_ep													
       , cova_deductible														
	   , case when cova_deductible=0 then 0 else ln(cast(cova_deductible as float)) end log_cova_deductible													
       , case 														
         when ceil(public.removenotnumeric(cova_limit) / 100000) * 100000<=1000000 then ceil(public.removenotnumeric(cova_limit) / 100000) * 100000														
         when ceil(public.removenotnumeric(cova_limit) / 100000) * 100000>1000000 and ceil(public.removenotnumeric(cova_limit) / 100000) * 100000<=1200000 then 1200000														
         else 1300000														
         end cova_limit														
	   , case when cova_limit=0 then 0 else ln(cast(cova_limit as float)) end log_cova_limit													
       , cova_ic_nc_water														
	   , case when cova_ic_nc_water>0 then 1 else 0 end HasClaim													
       , cova_il_nc_water	   													
       , case when cova_il_nc_water=0 then 0 else ln(cast(cova_il_nc_water as float)) end log_cova_il_nc_water														
/*----------------------------------------Water Scores-----------------------------------------------*/	   													
       , water_risk_3_blk														
       , case when water_risk_3_blk=0 then 0 else ln(cast(water_risk_3_blk as float)) end log_water_risk_3_blk														
       , water_risk_fre_3_blk	   													
       , case when water_risk_fre_3_blk=0 then 0 else ln(cast(water_risk_fre_3_blk as float)) end log_water_risk_fre_3_blk														
	   , water_risk_sev_3_blk													
	   ,case when water_risk_sev_3_blk=0 then 0 else  ln(cast(water_risk_sev_3_blk as float)) end log_water_risk_sev_3_blk													
	   ,case													
	     when appl_fail_3_blk='Highest' then 5													
	     when appl_fail_3_blk='High' then 4													
	     when appl_fail_3_blk='Average' then 3			 										
	     when appl_fail_3_blk='Below Avg' then	2												
		 when appl_fail_3_blk='Elevated' then 1												
		 when appl_fail_3_blk='Low' then	0											
		 else -1												
	    end appl_fail_3_blk													
	   ,case													
	     when fixture_leak_3_blk='Highest' then 5													
	     when fixture_leak_3_blk='High' then 4													
	     when fixture_leak_3_blk='Average' then 3			 										
	     when fixture_leak_3_blk='Below Avg' then	2												
		 when fixture_leak_3_blk='Elevated' then 1												
		 when fixture_leak_3_blk='Low' then	0											
		 else -1												
	    end fixture_leak_3_blk													
	   ,case													
	     when pipe_froze_3_blk='Highest' then 5													
	     when pipe_froze_3_blk='High' then 4													
	     when pipe_froze_3_blk='Average' then 3			 										
	     when pipe_froze_3_blk='Below Avg' then	2												
		 when pipe_froze_3_blk='Elevated' then 1												
		 when pipe_froze_3_blk='Low' then	0											
		 else -1												
	    end pipe_froze_3_blk													
	   ,case													
	     when plumb_leak_3_blk='Highest' then 5													
	     when plumb_leak_3_blk='High' then 4													
	     when plumb_leak_3_blk='Average' then 3			 										
	     when plumb_leak_3_blk='Below Avg' then	2												
		 when plumb_leak_3_blk='Elevated' then 1												
		 when plumb_leak_3_blk='Low' then	0											
		 else -1												
	    end plumb_leak_3_blk													
	   ,case													
	     when rep_cost_3_blk='Highest' then 5													
	     when rep_cost_3_blk='High' then 4													
	     when rep_cost_3_blk='Average' then 3			 										
	     when rep_cost_3_blk='Below Avg' then	2												
		 when rep_cost_3_blk='Elevated' then 1												
		 when rep_cost_3_blk='Low' then	0											
		 else -1												
	    end rep_cost_3_blk													
	   ,case													
	     when ustructure_fail_3_blk='Highest' then 5													
	     when ustructure_fail_3_blk='High' then 4													
	     when ustructure_fail_3_blk='Average' then 3			 										
	     when ustructure_fail_3_blk='Below Avg' then	2												
		 when ustructure_fail_3_blk='Elevated' then 1												
		 when ustructure_fail_3_blk='Low' then	0											
		 else -1												
	    end ustructure_fail_3_blk													
	   ,case													
	     when waterh_fail_3_blk='Highest' then 5													
	     when waterh_fail_3_blk='High' then 4													
	     when waterh_fail_3_blk='Average' then 3			 										
	     when waterh_fail_3_blk='Below Avg' then	2												
		 when waterh_fail_3_blk='Elevated' then 1												
		 when waterh_fail_3_blk='Low' then	0											
		 else -1												
	    end waterh_fail_3_blk	 												
/*-------------------------------------------------------------------------------------------------*/	   	   												
       , loaddate														
/*-------------------------------------------------------------------------------------------------*/	   													
       , customer_cnt_active_policies														
	   , case 													
	     when customer_cnt_active_policies=1 then 1	   												
	     when customer_cnt_active_policies>  1     and customer_cnt_active_policies<=	10	then 	10										
	     when customer_cnt_active_policies>	10    and customer_cnt_active_policies<=	15	then 	15									
	     when customer_cnt_active_policies>	15    and customer_cnt_active_policies<=	20	then 	20									
	     when customer_cnt_active_policies>	20    and customer_cnt_active_policies<=	30	then 	30									
	     when customer_cnt_active_policies>	30    and customer_cnt_active_policies<=	40	then 	40									
	     when customer_cnt_active_policies>	40    and customer_cnt_active_policies<=	50	then 	50									
	     when customer_cnt_active_policies>	50    and customer_cnt_active_policies<=	70	then 	70									
	     when customer_cnt_active_policies>	70    and customer_cnt_active_policies<=	90	then 	90									
	     when customer_cnt_active_policies>	90    and customer_cnt_active_policies<=	110	then 	110									
	     when customer_cnt_active_policies>	110   and customer_cnt_active_policies<=	120	then 	120									
	     when customer_cnt_active_policies>	120   and customer_cnt_active_policies<=	130	then 	130									
	     when customer_cnt_active_policies>	130	  then 	150										
        end customer_cnt_active_policies_binned														
from wc_PROPERTY_MODELDATA														
where pipe_froze_3_blk>-1 ;														
														
														
														
														
														
comment on table kdlab.property_water_claims_non_cat_fs_v5 is 'Property water claims frequency and severity consolidated training and testing data. No UT,  removed some extrim values records. Created 08/07/2020';														
														
													
																									
