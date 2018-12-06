#!/usr/bin/env bash


hive -e "
set hivevar:cid=131;
CREATE TABLE  amin.taxonomy_${hivevar:cid}
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
STORED AS TEXTFILE
as
select
distinct googletaxonomyid
from annotation.unfiltered_skus_subset
where cid=${hivevar:cid}
and  dt>='2018-01-05' and dt<='2018-01-18';
"

hive -e "
set hivevar:cid=101;
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;
add jar hdfs:/lib/udf.jar;
add jar hdfs:/lib/stemmer-udf.jar;
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
select searchterm,
product_pid,
listing_pid
from
(
select searchterms as searchterm,
product_pid,
listing_pid
from
annotation.product_to_listing
where dt>='2018-01-05' and dt<='2018-01-18'
and cid=${hivevar:cid}
and unix_timestamp(product_requesttime)-unix_timestamp(listing_requesttime)<30
)
a_one
LEFT JOIN
(
select querystring['searchterms'] as st, split(querystring['impressionpid'],':')[0] as pid
from default.beacon_logs_partitioned_daily
where action = 'click' and querystring['hlpt'] = 'S'
and dt>='2018-01-05' and dt<='2018-01-18'
and cid = ${hivevar:cid}
)
a_two
ON(a_one.searchterm=a_two.st and a_one.listing_pid=a_two.pid)
where (a_two.st is null)
)
a
JOIN
(
select beacon_pid, sku_skuid as sku
from annotation.beacon_to_sku
where dt>='2018-01-05' and dt<='2018-01-18'
and cid=${hivevar:cid}
)
c
ON(a.product_pid=c.beacon_pid)
JOIN
(
select
skuid, name, shortdescription, brand, taxonomyid, advertiserid
from annotation.unfiltered_skus_subset
where cid=${hivevar:cid}
and dt>='2018-01-05' and dt<='2018-01-18'
and  brand<>'' and taxonomyid<>-1
group by skuid, name, shortdescription, brand, taxonomyid, advertiserid
)
b
ON (c.sku=b.skuid)
DISTRIBUTE BY rand()
SORT BY rand();
"


hive -e "
set hivevar:cid=101;
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;

add jar hdfs:/lib/udf.jar;
add jar hdfs:/lib/stemmer-udf.jar;
create temporary function decodesafe as 'com.hooklogic.udf.UDFDecodeSafe';
create temporary function stemmer as 'udfs.HiveStemmer';
use amin;
DROP TABLE  IF EXISTS  amin.negative_training_samples_query_productname_stemmed_${hivevar:cid};
CREATE TABLE  amin.negative_training_samples_query_productname_stemmed_${hivevar:cid}
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
STORED AS TEXTFILE
as
select
stemmer(trim(lower(name)),${hivevar:cid}, 'en-US') as name,
stemmer(trim(lower(shortdescription)),${hivevar:cid}, 'en-US')  as description,
stemmer(trim(lower(brand)),${hivevar:cid}, 'en-US') as brand,
taxonomyid
from
(
select
product_pid,
listing_pid
from
annotation.product_to_listing
where dt>='2018-01-05' and dt<='2018-01-18'
and cid=${hivevar:cid}
)
a
JOIN
(
select beacon_pid, sku_skuid as sku
from annotation.beacon_to_sku
where dt>='2018-01-05' and dt<='2018-01-18'
and cid=${hivevar:cid}
)
c
ON(a.product_pid=c.beacon_pid)
JOIN
(
select
skuid, name, brand, taxonomyid, advertiserid, shortdescription
from annotation.unfiltered_skus_subset
where cid=${hivevar:cid}
and  dt>='2018-01-05' and dt<='2018-01-18'
and  brand<>'' and taxonomyid<>-1
group by skuid, name, brand, taxonomyid, advertiserid, shortdescription
)
b
ON (c.sku=b.skuid)
DISTRIBUTE BY rand()
SORT BY rand();
"


hive -e "
set hivevar:cid=131;
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;
add archives hdfs:///user/amantrach/Python.zip;
add file ../src/eval.py;
add file ../resources/models/cid-131-letters-328000.index;
add file ../resources/models/cid-131-letters-328000.meta;
add file ../resources/models/cid-131-letters-328000.data-00000-of-00001;
CREATE TABLE amin.positive_training_samples_query_productname_predictions_${hivevar:cid}
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
STORED AS TEXTFILE
as
SELECT
TRANSFORM (raw_searchterm,raw_product_name)
USING './bin/python eval.py cid-131-letters-328000'
AS (raw_searchterm,raw_product_name, score)
FROM
(
select * from
amin.positive_training_samples_query_productname_${hivevar:cid}
limit 100
)a ;
"






hive -e "
set hivevar:cid=131;
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;
add jar hdfs:/lib/udf.jar;
create temporary function decodesafe as 'com.hooklogic.udf.UDFDecodeSafe';
use amin;
DROP TABLE  IF EXISTS  amin.positive_test_samples_query_productname_${hivevar:cid};
CREATE TABLE  amin.positive_test_samples_query_productname_${hivevar:cid}
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
STORED AS TEXTFILE
as
select
decodesafe(searchterm) as raw_searchterm,
name as raw_product_name,
skuid
from
(
select searchterms as searchterm,
product_pid,
listing_pid
from
annotation.product_to_listing
where dt='2018-01-19'
and cid=${hivevar:cid}
)
a
JOIN
(
select beacon_pid, sku_skuid as sku
from annotation.beacon_to_sku
where dt='2018-01-19'
and cid=${hivevar:cid}
)
c
ON(a.product_pid=c.beacon_pid)
JOIN
(
select
skuid, name, shortdescription, brand, taxonomyid, advertiserid
from annotation.unfiltered_skus_subset
where cid=${hivevar:cid}
and dt='2018-01-19'
and  brand<>'' and taxonomyid<>-1
group by skuid, name, shortdescription, brand, taxonomyid, advertiserid
)
b
ON (c.sku=b.skuid)
DISTRIBUTE BY rand()
SORT BY rand();

"
hive -e "
set hivevar:cid=131;
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;
add jar hdfs:/lib/udf.jar;
create temporary function decodesafe as 'com.hooklogic.udf.UDFDecodeSafe';
use amin;
DROP TABLE  IF EXISTS  amin.positive_test_samples_query_productname_${hivevar:cid};
CREATE TABLE  amin.positive_test_samples_query_productname_${hivevar:cid}
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
STORED AS TEXTFILE
as
select
stemmer(trim(lower(decodesafe(searchterm))),${hivevar:cid}, 'en-US') as stemmed_searchterm,
stemmer(trim(lower(name)),${hivevar:cid}, 'en-US') as stemmed_product_name,
skuid
from
(
select searchterms as searchterm,
product_pid,
listing_pid
from
annotation.product_to_listing
where dt='2018-01-19'
and cid=${hivevar:cid}
)
a
JOIN
(
select beacon_pid, sku_skuid as sku
from annotation.beacon_to_sku
where dt='2018-01-19'
and cid=${hivevar:cid}
)
c
ON(a.product_pid=c.beacon_pid)
JOIN
(
select
skuid, name, shortdescription, brand, taxonomyid, advertiserid
from annotation.unfiltered_skus_subset
where cid=${hivevar:cid}
and dt='2018-01-19'
and  brand<>'' and taxonomyid<>-1
group by skuid, name, shortdescription, brand, taxonomyid, advertiserid
)
b
ON (c.sku=b.skuid)
DISTRIBUTE BY rand()
SORT BY rand();

"



hive -e "
set hivevar:cid=168;
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;
add jar hdfs:/lib/udf.jar;
add jar hdfs:/lib/stemmer-udf.jar;



create temporary function decodesafe as 'com.hooklogic.udf.UDFDecodeSafe';
create temporary function stemmer as 'udfs.HiveStemmer';

use amin;
DROP TABLE  IF EXISTS  amin.positive_training_samples_query_productname_descr_brand_category_stemmed_unique_${hivevar:cid};
CREATE TABLE  amin.positive_training_samples_query_productname_descr_brand_category_stemmed_unique_${hivevar:cid}
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
STORED AS TEXTFILE
as
select
stemmer(trim(lower(decodesafe(searchterm))),${hivevar:cid}, 'en-US') as stemmed_searchterm,
stemmer(trim(lower(max(name))),${hivevar:cid}, 'en-US') as stemmed_product_name,
stemmer(trim(lower(max(shortdescription))),${hivevar:cid}, 'en-US') as shortdescription,
stemmer(trim(lower(max(brand))),${hivevar:cid}, 'en-US') as brand,
max(taxonomyid)
from
(
select searchterm,
product_pid,
listing_pid
from
(
select searchterms as searchterm,
product_pid,
listing_pid
from
annotation.product_to_listing
where  dt>='2018-01-05' and dt<='2018-01-18'
and cid=${hivevar:cid}
and unix_timestamp(product_requesttime)-unix_timestamp(listing_requesttime)<30
)
a_one
LEFT JOIN
(
select querystring['searchterms'] as st, split(querystring['impressionpid'],':')[0] as pid
from default.beacon_logs_partitioned_daily
where action = 'click' and querystring['hlpt'] = 'S'
and dt>='2018-01-05' and dt<='2018-01-18'
and cid = ${hivevar:cid}
)
a_two
ON(a_one.searchterm=a_two.st and a_one.listing_pid=a_two.pid)
where (a_two.st is null)
)
a
JOIN
(
select beacon_pid, sku_skuid as sku
from annotation.beacon_to_sku
where  dt>='2018-01-05' and dt<='2018-01-18'
and cid=${hivevar:cid}
)
c
ON(a.product_pid=c.beacon_pid)
JOIN
(
select
skuid, name, shortdescription, brand, taxonomyid, advertiserid
from annotation.unfiltered_skus_subset
where cid=${hivevar:cid}
and  dt>='2018-01-05' and dt<='2018-01-18'
and  brand<>'' and taxonomyid<>-1
group by skuid, name, shortdescription, brand, taxonomyid, advertiserid
)
b
ON (c.sku=b.skuid)
group by
stemmer(trim(lower(decodesafe(searchterm))),${hivevar:cid}, 'en-US'), skuid
having count(*)>5
DISTRIBUTE BY rand()
SORT BY rand();
"

hive -e
"
set hivevar:cid=131;
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;
add jar hdfs:/lib/udf.jar;
add jar hdfs:/lib/stemmer-udf.jar;
set hivevar:extractionday='2018-01-20';
set hivevar:name=validation;
create table amin.positive_training_samples_query_productname_descr_brand_category_stemmed_unique_cross_${hivevar:name}_${hivevar:cid}
as
select
stemmed_searchterm,
stemmed_product_name,
shortdescription,
brand,
taxid
from
(
select
stemmed_searchterm
from
amin.positive_training_samples_query_productname_descr_brand_category_stemmed_unique_${hivevar:name}_${hivevar:cid}
group by stemmed_searchterm
) a
CROSS JOIN
(
select
stemmed_product_name, shortdescription, brand, `_c4` taxid
from
amin.positive_training_samples_query_productname_descr_brand_category_stemmed_unique_${hivevar:name}_${hivevar:cid}
group by stemmed_product_name, shortdescription, brand, `_c4`
) b;
"






hive -e "
set hivevar:cid=131;
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;
add jar hdfs:/lib/udf.jar;
add jar hdfs:/lib/stemmer-udf.jar;

create temporary function decodesafe as 'com.hooklogic.udf.UDFDecodeSafe';
create temporary function stemmer as 'udfs.HiveStemmer';
use amin;
DROP TABLE  IF EXISTS  amin.negative_training_samples_query_productname_descr_brand_categgory_stemmed_unique_${hivevar:cid};
CREATE TABLE  amin.negative_training_samples_query_productname_descr_brand_categgory_stemmed_unique_${hivevar:cid}
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
STORED AS TEXTFILE
as
select
stemmer(trim(lower(name)),${hivevar:cid}, 'en-US') as stemmed_product_name,
stemmer(trim(lower(shortdescription)),${hivevar:cid}, 'en-US') as shortdescription,
stemmer(trim(lower(brand)),${hivevar:cid}, 'en-US') as brand,
taxonomyid
from annotation.unfiltered_skus_subset
where cid=${hivevar:cid}
and  dt>='2018-01-05' and dt<='2018-01-18'
and  brand<>'' and taxonomyid<>-1
group by skuid, name, brand, taxonomyid, advertiserid, shortdescription
DISTRIBUTE BY rand()
SORT BY rand();
"

hive -e "

set hivevar:cid=131;
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;
add jar hdfs:/lib/udf.jar;
add jar hdfs:/lib/stemmer-udf.jar;

create temporary function decodesafe as 'com.hooklogic.udf.UDFDecodeSafe';
create temporary function stemmer as 'udfs.HiveStemmer';

create table amin.keywordmodel_log_withrawkeyword_stemmed
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
as
select
stemmed_keyword,
stemmer(trim(lower(name)),${hivevar:cid}, 'en-US') as name,
stemmer(trim(lower(description)),${hivevar:cid}, 'en-US')  as description,
stemmer(trim(lower(brand)),${hivevar:cid}, 'en-US') as brand ,
in_keyword_model,
full_skuid
from amin.keywordmodel_log a
JOIN
(
select * from amin.rawquery_to_stemmed
) b
ON (a.stemmed_keyword=b.stemmed_searchterm);
"



hive -e "
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;

add jar hdfs:/lib/udf.jar;
add jar hdfs:/lib/stemmer-udf.jar;
create temporary function decodesafe as 'com.hooklogic.udf.UDFDecodeSafe';
create temporary function stemmer as 'udfs.HiveStemmer';
use amin;
DROP TABLE  IF EXISTS  amin.negative_training_samples_query_productname_stemmed_all;
CREATE TABLE  amin.negative_training_samples_query_productname_stemmed_notfrom_all
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
STORED AS TEXTFILE
as
select
stemmer(trim(lower(name)),a.cid, 'en-US') as name,
stemmer(trim(lower(shortdescription)),a.cid, 'en-US')  as description,
stemmer(trim(lower(brand)),a.cid, 'en-US') as brand,
taxonomyid,
a.cid
from
(
select
product_pid,
listing_pid,
cid
from
annotation.product_to_listing
where dt>='2018-01-05' and dt<='2018-01-18'
)
a
JOIN
(
select cid from model.hookusercompanies
where culture='en-US'
)
e
ON (a.cid=e.cid)
JOIN
(
select beacon_pid, sku_skuid as sku, cid
from annotation.beacon_to_sku
where dt>='2018-01-05' and dt<='2018-01-18'
)
c
ON(a.product_pid=c.beacon_pid and a.cid=c.cid)
JOIN
(
select
skuid, name, brand, taxonomyid, advertiserid, shortdescription, cid
from annotation.unfiltered_skus_subset
where dt>='2018-01-05' and dt<='2018-01-18'
and  brand<>'' and taxonomyid<>-1
group by skuid, name, brand, taxonomyid, advertiserid, shortdescription, cid
)
b
ON (c.sku=b.skuid and c.cid=b.cid)
DISTRIBUTE BY rand()
SORT BY rand();
"

