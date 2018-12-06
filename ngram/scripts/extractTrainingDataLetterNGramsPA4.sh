#!/usr/bin/env bash
hive -e "
set hivevar:cid=131;
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;
add jar /home/a.mantrach/lib/udf.jar;
add jar  /home/a.mantrach/lib/stemmer-udf.jar;
create temporary function decodesafe as 'com.hooklogic.udf.UDFDecodeSafe';
create temporary function stemmer as 'udfs.HiveStemmer';

use amin;
DROP TABLE  IF EXISTS  amin.positive_training_samples_query_productname_stemmed_${hivevar:cid};
CREATE TABLE  amin.positive_training_samples_query_productname_stemmed_${hivevar:cid}
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
STORED AS TEXTFILE
as
select
stemmer(trim(lower(decodesafe(searchterm))),${hivevar:cid}, 'en-US') as stemmed_searchterm,
stemmer(trim(lower(name)),${hivevar:cid}, 'en-US') as name,
stemmer(trim(lower(shortdescription)),${hivevar:cid}, 'en-US')  as description,
stemmer(trim(lower(brand)),${hivevar:cid}, 'en-US') as brand,
taxonomyid
from
(
--select searchterm,
--product_pid,
--listing_pid
--from
--(
select keyword as searchterm,
product_id as product_pid,
search_id listing_pid
from
cbskey.product_searches
where date>='2018-06-01' and date<='2018-06-01'
and media_source_id=${hivevar:cid}
and product_event_time-search_event_time<30
--)
--a_one
--LEFT JOIN
--(
--select decodesafe(keyword) as st, split(impression_ext_id,':')[0] as pid
--from cbs_datacontract.clicks
--where batch_date>='2018-06-01' and batch_date<='2018-06-01'
--and page_type = 'S' and
--media_source_id=131
--)
--a_two
--ON(a_one.searchterm=a_two.st and a_one.listing_pid=a_two.pid)
--where (a_two.st is null)
)
a
JOIN
(
select product_id beacon_pid, skuid as sku
from cbskey.beacon_sku
where date>='2018-06-01' and date<='2018-06-01'
and cid=${hivevar:cid}
)
c
ON(a.product_pid=c.beacon_pid)
JOIN
(
select
sku_id skuid, name, description shortdescription, brand, taxonomyid
from cbskey.unfiltered_sku_subset
where media_source_id=${hivevar:cid}
and date>='2018-06-01' and date<='2018-06-01'
and  brand<>'' and taxonomyid<>-1
group by sku_id, name, description, brand, taxonomyid
)
b
ON (c.sku=b.skuid)
DISTRIBUTE BY rand()
SORT BY rand();
"



select keyword,
product_ext_id as product_id,
search_id
from
cbskey.product_searches
where date='2018-03-05'
and media_source_id=131
and product_event_time-search_event_time<30
limit 100


select searchterm,
product_pid,
listing_pid
from
(
select keyword as searchterm,
product_ext_id as product_pid,
search_id as listing_pid
from
cbskey.product_searches
where date='2018-03-05'
and media_source_id=131
and product_event_time-search_event_time<30
)
a_one
JOIN
(
select keyword as st, split(ad_targeting_ext_id,':')[0] as pid
from cbs_datacontract.clicks
where batch_date='2018-03-05'
and page_type = 'S' and
media_source_id=131
)
a_two
ON(a_one.searchterm=a_two.st and a_one.listing_pid=a_two.pid);