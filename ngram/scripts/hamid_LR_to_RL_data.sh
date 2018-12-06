#!/usr/bin/env bash
hive -e "
set hivevar:cid=131;
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;
set hivevar:name=training_LR_to_RL;
drop table if exists amin.sponsored_${hivevar:name}_impressions_${hivevar:cid};
create table amin.sponsored_${hivevar:name}_impressions_${hivevar:cid}
as
select impressions.pageguid, st, impressions.rank, impressions.pid, skuid, (case when clicks.pid is null then 0 else 1 end) as click
from
(
select querystring['pageguid'] as pageguid, querystring['searchterms'] as st, querystring['pid'] as pid, querystring['rank'] as rank,
querystring['productsku'] as skuid
from default.beacon_logs_partitioned_daily
where action = 'imp' and querystring['hlpt'] = 'S'
and  querystring['rt']<>'IAB'
and dt>='2018-01-05' and dt<='2018-01-18'
and cid = ${hivevar:cid}
group by querystring['pageguid'], querystring['searchterms'], querystring['pid'], querystring['rank'], querystring['productsku']
)
impressions
LEFT JOIN
(
select querystring['impressionpid'] as pid
from default.beacon_logs_partitioned_daily
where action = 'click' and querystring['hlpt'] = 'S'
and  querystring['rt']<>'IAB'
and dt>='2018-01-05' and dt<='2018-01-18'
and cid = ${hivevar:cid}
)
clicks
ON (impressions.pid=clicks.pid);
"

hive -e "
set hivevar:name=training_LR_to_RL;
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
set hivevar:name=training_LR_to_RL;
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
set hivevar:extractionday=2018-01-19;
set hivevar:name=training_LR_to_RL;
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
and dt='where cid=${hivevar:cid}
and dt>='2018-01-05' and dt<='2018-01-18'
and  brand<>'' and taxonomyid<>-1
group by skuid
)
b
ON(a.skuid=b.skuid)
group by pageguid, st, rank, click, name, shortdescription, brand, a.skuid, taxonomyid
order by pageguid, st, rank;
2018-01-19'
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
set hivevar:name=training_LR_to_RL;
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
