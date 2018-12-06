set hivevar:cid=131;
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;

drop table  if exists amin.organics_searchterms_forsponsoredskus_${hivevar:cid};
create table amin.organics_searchterms_forsponsoredskus_${hivevar:cid}
as
select searchterms, sku, organics.dt, count(*) nbclicks
from
(
select beacon_pid , sku_skuid sku, dt
from annotation.beacon_to_sku
where dt>='2018-01-05' and dt<='2018-01-18'
and cid=${hivevar:cid}
group by beacon_pid, sku_skuid, dt
)
organics_sku
JOIN
(
select querystring['productsku'] skuid, dt
from default.beacon_logs_partitioned_daily
where action = 'imp' and querystring['hlpt'] = 'S'
and dt>='2018-01-05' and dt<='2018-01-18'
and cid = ${hivevar:cid}
group by querystring['productsku'], dt
)
sponsored_skus
ON(sponsored_skus.skuid=organics_sku.sku and sponsored_skus.dt=organics_sku.dt)
JOIN
(
select searchterms,
product_pid,
dt
from
annotation.product_to_listing
where dt>='2018-01-05' and dt<='2018-01-18'
and cid=${hivevar:cid}
and unix_timestamp(product_requesttime)-unix_timestamp(listing_requesttime)<60
and is_direct=True
group by searchterms, product_pid, dt
)
organics
ON
(organics.product_pid=organics_sku.beacon_pid and organics.dt=organics_sku.dt)
group by searchterms, sku, organics.dt
having count(*)>2;



drop table if exists amin.sponsored_searchterms_forsponsoredskus_${hivevar:cid};
create table amin.sponsored_searchterms_forsponsoredskus_${hivevar:cid}
as
select searchterms
from
(
select  sku_skuid sku
from annotation.beacon_to_sku
where dt>='2018-01-05' and dt<='2018-01-18'
and cid=${hivevar:cid}
group by sku_skuid
)
organics_sku
JOIN
(
select
querystring['searchterms'] searchterms,
querystring['productsku'] skuid,
dt
from default.beacon_logs_partitioned_daily
where action = 'imp' and querystring['hlpt'] = 'S'
and dt>='2018-01-05' and dt<='2018-01-18'
and cid = ${hivevar:cid}
group by querystring['searchterms'], querystring['productsku']
)
sponsored_skus
ON(sponsored_skus.skuid=organics_sku.sku)
group by searchterms;



drop table if exists amin.organics_exclusive_searchterms_forsponsoredskus_${hivevar:cid};
create table amin.organics_exclusive_searchterms_forsponsoredskus_${hivevar:cid}
as
select
a.searchterms, a.sku, a.nbclicks, a.dt from
amin.organics_searchterms_forsponsoredskus_${hivevar:cid} a
LEFT JOIN
amin.sponsored_searchterms_forsponsoredskus_${hivevar:cid} b
ON (a.searchterms=b.searchterms)
where b.searchterms is null;


drop table if exists amin.organicssearches_cnt;
create table amin.organicssearches_cnt
as
select searchterms, count(searchterms) countsearches
from
annotation.product_to_listing
where dt>='2018-01-05' and dt<='2018-01-18'
and cid=${hivevar:cid}
group by searchterms;



set hivevar:cid=131;
set hive.map.aggr=false;
set hive.auto.convert.join=false;
set dfs.namenode.acls.enabled=false;
drop table if exists amin.organics_exclusive_searchterms_forsponsoredskus_with_clickcount_${hivevar:cid};
create table amin.organics_exclusive_searchterms_forsponsoredskus_with_clickcount_${hivevar:cid}
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
as
select a.searchterms, a.sku, a.nbclicks, a.dt, countsearches
from
amin.organics_exclusive_searchterms_forsponsoredskus_${hivevar:cid} a
join
amin.organicssearches_cnt b
where (a.searchterms=b.searchterms)
order by countsearches desc;






drop table if exists amin.organics_exclusive_searchterms_forsponsoredskus_with_clickcount_andskucount_${hivevar:cid};
create table amin.organics_exclusive_searchterms_forsponsoredskus_with_clickcount_andskucount_${hivevar:cid}
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
as
select searchterms, countsearches, collect_set(sku) skuset, collect_set(nbclicks) clickset, sum(nbclicks) allclicks
from
amin.organics_exclusive_searchterms_forsponsoredskus_with_clickcount_${hivevar:cid}
group by searchterms, dt, countsearches
order by countsearches desc;



select sum(allclicks) from amin.organics_exclusive_searchterms_forsponsoredskus_with_clickcount_andskucount_${hivevar:cid};


19989 clicks

select count(*) from
(
select
querystring['pid'] pid
from default.beacon_logs_partitioned_daily
where action = 'click' and querystring['hlpt'] = 'S'
and dt>='2018-01-05' and dt<='2018-01-18'
and cid = ${hivevar:cid}
group by querystring['pid']
) z;


168.341 >> clicks

select count(*)
from
(
select sku from
amin.organics_exclusive_searchterms_forsponsoredskus_${hivevar:cid}
group by sku
) l;


select count(*)
from
(
select searchterms
from
amin.organics_exclusive_searchterms_forsponsoredskus_${hivevar:cid}
group by searchterms
) l;

amin.organics_exclusive_searchterms_forsponsoredskus_${hivevar:cid}

select count(*)
from
(
select searchterms
from
amin.organics_searchterms_forsponsoredskus_${hivevar:cid}
group by searchterms
) l;


amin.organics_searchterms_forsponsoredskus_${hivevar:cid}
5297