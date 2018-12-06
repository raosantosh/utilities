#!/usr/bin/env bash

hadoop fs -mkdir aminhivedb/annotations
hadoop fs -copyFromLocal /tmp/annotations_40K.csv aminhivedb/annotations

hive -e "
use amin;
DROP TABLE IF EXISTS annotations;
CREATE EXTERNAL TABLE  annotations(keyword string, skukey string,  rating string)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/user/a.mantrach/aminhivedb/annotations/';
"

hive -e"
set hivevar:cid=131;
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;



use amin;
drop table if exists annotations_full;
create table annotations_full
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
STORED AS TEXTFILE
as
select
keyword,
name,
brand,
rating,
sku_id
from
(
select
keyword,
split(skukey,'-')[0] as a_cid,
split(skukey,'-')[2] as a_skuid,
rating
from amin.annotations
where keyword is not null and keyword!=''
group by keyword, skukey, rating
)
a
JOIN
(
select sku_id, name, brand, media_source_id
from
cbskey.unfiltered_sku_subset
where date='2018-05-10'
and name is not null
and brand is not null
and name!=''
and brand!=''
and sku_id is not null
group by sku_id, name, brand, media_source_id
) b
ON (a.a_cid=b.media_source_id and a.a_skuid=b.sku_id);
"

hive -e
"
add jar /tmp/udf.jar;
add jar /tmp/stemmer-udf.jar;
create temporary function decodesafe as 'com.hooklogic.udf.UDFDecodeSafe';
create temporary function stemmer as 'udfs.HiveStemmer';
set hivevar:cid=131;
use amin;
drop table if exists annotations_full_stemmed;
create table annotations_full_stemmed
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
STORED AS TEXTFILE
as
select
stemmer(trim(lower(decodesafe(keyword))),${hivevar:cid}, 'en-US') stemmed_keyword,
stemmer(trim(lower(decodesafe(name))),${hivevar:cid}, 'en-US') stemmed_name,
stemmer(trim(lower(decodesafe(brand))),${hivevar:cid}, 'en-US') stemmed_brand,
keyword,
name,
brand,
sku_id,
rating
from
annotations_full
where trim(lower(decodesafe(keyword))) is not null
and trim(lower(decodesafe(name))) is not null
and trim(lower(decodesafe(brand))) is not null;
"




select
stemmer('alpha',${hivevar:cid}, 'en-US'),
split(skukey,'-')[0] as a_cid,
split(skukey,'-')[1] as a_skuid,
skukey
from amin.annotations
limit 200;


select
 stemmed_keyword,
stemmer(trim(lower(decodesafe(name))),${hivevar:cid}, 'en-US') stemmed_name,
stemmer(trim(lower(decodesafe(brand))),${hivevar:cid}, 'en-US') stemmed_brand,
keyword,
name,
brand,
rating,
sku_id
from
(
select
keyword,
split(skukey,'-')[0] as a_cid,
split(skukey,'-')[2] as a_skuid,
rating
from amin.annotations
where keyword is not null and keyword!=''
group by keyword, skukey, rating