hive -e "
set hivevar:cid=131;
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;

drop table if exists amin.sponsored_impressions_${hivevar:cid};
create table amin.sponsored_impressions_${hivevar:cid}
as
select impressions.pageguid, st, impressions.rank, impressions.pid, skuid, (case when clicks.pid is null then 0 else 1 end) as click
from
(
select querystring['pageguid'] as pageguid, querystring['searchterms'] as st, querystring['pid'] as pid, querystring['rank'] as rank,
querystring['productsku'] as skuid
from default.beacon_logs_partitioned_daily
where action = 'imp' and querystring['hlpt'] = 'S'
and  querystring['rt']<>'IAB'
and dt='2018-01-19'
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
and dt='2018-01-19'
and cid = ${hivevar:cid}
)
clicks
ON (impressions.pid=clicks.pid);
"

hive -e "
drop table if exists amin.sponsored_impressions_withoneclickatleast_${hivevar:cid};
create table amin.sponsored_impressions_withoneclickatleast_${hivevar:cid}
as
select pageguid, st
from
amin.sponsored_impressions_${hivevar:cid}
group by pageguid, st
having count(*)>2 and sum(click)>0;
"

hive -e "
drop table if exists amin.sponsored_clicked_impressions_${hivevar:cid};
create table amin.sponsored_clicked_impressions_${hivevar:cid}
as
select a.pageguid, a.st, rank, pid, click, skuid
from
amin.sponsored_impressions_${hivevar:cid} a
JOIN
amin.sponsored_impressions_withoneclickatleast_${hivevar:cid} b
where (a.pageguid =b.pageguid and a.st=b.st);
"


hive -e "
set hivevar:cid=131;
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;
drop table if exists amin.sponsored_clicked_impressions_withskus_${hivevar:cid};
create table amin.sponsored_clicked_impressions_withskus_${hivevar:cid}
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
from amin.sponsored_clicked_impressions_${hivevar:cid} a
JOIN
(
select
skuid, max(name) name, max(shortdescription) shortdescription, max(brand) brand, max(taxonomyid) taxonomyid
from annotation.unfiltered_skus_subset
where cid=${hivevar:cid}
and dt='2018-01-19'
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
drop table if exists amin.sponsored_clicked_impressions_withskus_stemmed_${hivevar:cid};
create table amin.sponsored_clicked_impressions_withskus_stemmed_${hivevar:cid}
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
amin.sponsored_clicked_impressions_withskus_${hivevar:cid};
"




hive -e "
drop table if exists amin.sponsored_clicked_impressions_withskus_stemmed_validation_131;
create external table amin.sponsored_clicked_impressions_withskus_stemmed_validation_131
(
pageguid string,
st string,
rank int,
click int,
name string,
shortdescription string,
brand string,
skuid int,
taxonomyid int
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
STORED AS TEXTFILE
location '/user/a.mantrach/aminhivedb/sponsored_clicked_impressions_withskus_stemmed_validation_131';

"

create table amin.sponsored_clicked_impressions_withskus_stemmed_validation_131_with_auction_infos
as
select *
from
amin.sponsored_clicked_impressions_withskus_stemmed_validation_131 a
JOIN
(
select ad_targeting_id, skus
from  glup.glup_cbs_auction_events
where day='2018-01-19'
and media_source_id=131
and host_platform='US'
and page_type='S'
) b
where(a.pageguid=b.ad_targeting_id);

from tbl


select page_type, count(*) from
glup.glup_cbs_auction_events
where day='2018-01-19'
and hour=0
and media_source_id=131
and host_platform='US'
group  by page_type;


show partitions glup.glup_cbs_auction_events;

select ad_targeting_id, skus
from  glup.glup_cbs_auction_events
where day<='2018-01-21' and day>='2018-01-18'
and ad_targeting_id='001d0faa-2f75-47c9-a16d-c17c8fc91e1b';

select *
from
glup_cbs_auction_events
where day='2018-01-18'
and media_source_id=131
and host_platform='US'
and auction_guid='0028197a-85a8-4425-a521-b835563ecad9';



hive -e "
create external table amin.annotations_stemmed_all
(keyword string, name string, description string, brand string, global_taxonomy_id string, in_memo string, annotation string)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
LOCATION '/user/amantrach/aminhivedb/all_annotations.json_minimodel';

use amin;
set hivevar:cid=131;
add jar hdfs:/lib/stemmer-udf.jar;
create temporary function stemmer as 'udfs.HiveStemmer';

DROP TABLE  IF EXISTS  amin.annotations_fully_stemmed_all;
CREATE TABLE amin.annotations_fully_stemmed_all
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
as
select
stemmer(trim(lower(keyword)),${hivevar:cid}, 'en-US') as keyword,
stemmer(trim(lower(name)),${hivevar:cid}, 'en-US') as name,
stemmer(trim(lower(description)),${hivevar:cid}, 'en-US') as description,
stemmer(trim(lower(brand)),${hivevar:cid}, 'en-US') as brand,
global_taxonomy_id,
in_memo,
annotation
from
amin.annotations_stemmed_all;
"



