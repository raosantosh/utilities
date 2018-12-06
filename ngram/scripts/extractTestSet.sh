#!/usr/bin/env bash

hive -e "
set hivevar:cid=131;
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;
set hivevar:extractionday='2018-01-19';
set hivevar:name=validation;
drop table if exists amin.sponsored_${hivevar:name}_impressions_${hivevar:cid};
create table amin.sponsored_${hivevar:name}_impressions_${hivevar:cid}
as
select impressions.pageguid, st, impressions.rank, impressions.pid, skuid, (case when clicks.pid is null then 0 else 1 end) as click
from
(
select querystring['aucid'] as pageguid, querystring['searchterms'] as st, querystring['pid'] as pid, querystring['rank'] as rank,
querystring['productsku'] as skuid
from default.beacon_logs_partitioned_daily
where action = 'imp' and querystring['hlpt'] = 'S'
and  querystring['rt']<>'IAB'
and dt=${hivevar:extractionday}
and cid = ${hivevar:cid}
group by querystring['aucid'], querystring['searchterms'], querystring['pid'], querystring['rank'], querystring['productsku']
)
impressions
LEFT JOIN
(
select querystring['impressionpid'] as pid
from default.beacon_logs_partitioned_daily
where action = 'click' and querystring['hlpt'] = 'S'
and  querystring['rt']<>'IAB'
and dt=${hivevar:extractionday}
and cid = ${hivevar:cid}
)
clicks
ON (impressions.pid=clicks.pid);
"

hive -e "
set hivevar:name=validation;
drop table if exists amin.sponsored_impressions_withoneclickatleast_${hivevar:name}_${hivevar:cid};
create table amin.sponsored_impressions_withoneclickatleast_${hivevar:name}_${hivevar:cid}
as
select pageguid, st
from
amin.sponsored_${hivevar:name}_impressions_${hivevar:cid}
group by pageguid, st
having count(*)>2 and sum(click)>0;
"

hive -e "
set hivevar:name=validation;
drop table if exists amin.sponsored_clicked_impressions_${hivevar:name}_${hivevar:cid};
create table amin.sponsored_clicked_impressions_${hivevar:name}_${hivevar:cid}
as
select a.pageguid, a.st, rank, pid, click, skuid
from
amin.sponsored_${hivevar:name}_impressions_${hivevar:cid} a
JOIN
amin.sponsored_impressions_withoneclickatleast_${hivevar:name}_${hivevar:cid} b
where (a.pageguid =b.pageguid and a.st=b.st);
"


hive -e "
set hivevar:cid=131;
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;
set hivevar:extractionday='2018-01-19';
set hivevar:name=validation;
drop table if exists amin.sponsored_clicked_impressions_withskus_${hivevar:name}_${hivevar:cid};
create table amin.sponsored_clicked_impressions_withskus_${hivevar:name}_${hivevar:cid}
as
select
pageguid,
st,
rank,
click,
name,
shortdescription,
brand,
a.skuid,
taxonomyid
from amin.sponsored_clicked_impressions_${hivevar:name}_${hivevar:cid} a
JOIN
(
select
skuid, max(name) name, max(shortdescription) shortdescription, max(brand) brand, max(taxonomyid) taxonomyid
from annotation.unfiltered_skus_subset
where cid=${hivevar:cid}
and dt=${hivevar:extractionday}
and  brand<>'' and taxonomyid<>-1
group by skuid
)
b
ON(a.skuid=b.skuid)
group by pageguid, st, rank, click, name, shortdescription, brand, a.skuid, taxonomyid
order by pageguid, st, rank;
"

hive -e "
add jar hdfs:/lib/udf.jar;
add jar hdfs:/lib/stemmer-udf.jar;
create temporary function decodesafe as 'com.hooklogic.udf.UDFDecodeSafe';
create temporary function stemmer as 'udfs.HiveStemmer';
set hivevar:cid=131;
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;
set hivevar:name=validation;
drop table if exists amin.sponsored_clicked_impressions_withskus_stemmed_${hivevar:name}_${hivevar:cid};
create table amin.sponsored_clicked_impressions_withskus_stemmed_${hivevar:name}_${hivevar:cid}
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
STORED AS TEXTFILE
as
select
pageguid,
stemmer(trim(lower(decodesafe(st))),${hivevar:cid}, 'en-US') as st,
rank,
click,
stemmer(trim(lower(name)),${hivevar:cid}, 'en-US') name,
stemmer(trim(lower(shortdescription)),${hivevar:cid}, 'en-US') shortdescription,
stemmer(trim(lower(brand)),${hivevar:cid}, 'en-US')  brand,
skuid,
taxonomyid
from
amin.sponsored_clicked_impressions_withskus_${hivevar:name}_${hivevar:cid};
"

drop table if exists  amin.sponsored_clicked_impressions_withskus_stemmed_validation_131;
create external table amin.sponsored_clicked_impressions_withskus_stemmed_validation_131
(pageguid string, st string, rank int, click int, name string, shortdescription string, brand string, skuid int, taxonomyid int)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
STORED AS TEXTFILE
location '/user/a.mantrach/aminhivedb/sponsored_clicked_impressions_withskus_stemmed_validation_131';


select



9c75445b-e700-43e3-86fe-7bf0874fecf8    165
b77aa94a-9eec-4cee-8658-6048bac48727    260
2826f60b-405c-46c4-89fe-e8e12b6e829d    165
a88ef8f4-2780-4750-b414-fb7be02473f8    256
32042885-8e12-46c7-9438-736678978224    302
ef6e5a6d-bfa8-4df4-9809-d884ee3e3425    302
cf744bcc-0569-47e5-bc30-136e800cc628    165
52c7b4f4-7364-4d9f-a833-ba2b45e4bc71    256
afdbdb7a-59ab-455a-9e2f-20640e33096c    270
4ff1a7ef-3a2c-4ea2-ac3e-eb5eaddc43ff    256


cd887e1f-0555-4e6d-a2e4-2fe6dfc72fbc    131
d6edacea-b71b-4eb4-82d3-f62bae64746e    131
7d09ea36-3097-4b57-8036-a205da4578e5    131
03a17c06-8503-40f8-8594-650ce8f12d51    131
98ac1355-7415-4a42-b92a-c10e5afe0120    131
8c64e016-f98f-40eb-9dc7-ffbb9c006487    131
e6e046b3-be26-4893-b829-f285f8991bbd    131
713b8c7c-4bc6-4c43-b87b-db91362fa0e1    131
f43332b0-532b-4d52-b942-5d4d219627f2    131
14c902a2-c771-4d28-9d82-02566bf7bef7    131


select dt, querystring
from
default.beacon_logs_partitioned_daily
where querystring['aucid']='14c902a2-c771-4d28-9d82-02566bf7bef7'
and dt='2018-01-19'
and cid = 131;

select *
from
glup_cbs_auction_events
where day='2018-01-19'
and media_source_id=131
and auction_guid='0011ee1c-82c6-4915-8b08-aa35b08adf86';

drop table if exists amin.sponsored_clicked_impressions_withskus_stemmed_validation_131_auctions;
create table amin.sponsored_clicked_impressions_withskus_stemmed_validation_131_auctions
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
STORED AS TEXTFILE
as
select
pageguid,
st,
rank,
click,
name,
shortdescription,
brand,
skuid,
taxonomyid,
skus[rank].quality_score  quality_score,
skus[rank].bid  bid,
skus[rank].cost  cost
from amin.sponsored_clicked_impressions_withskus_stemmed_validation_131 a
JOIN
(
select auction_guid, skus from
glup.glup_cbs_auction_events
where day='2018-01-19'
and  media_source_id=131
) b
ON (a.pageguid=b.auction_guid);
