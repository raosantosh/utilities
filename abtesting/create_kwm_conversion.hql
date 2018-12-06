SET hive.exec.reducers.max=40;
set hive.map.aggr=false;

add jar hdfs:/lib/udf-ds.jar;
create temporary function clientinfo as 'com.hooklogic.udf.UDFClientInfo';
create temporary function apache_timestamp as 'com.hooklogic.udf.UDFApacheTimestamp';

use default;

set dts=${startDate};
set dte=${endDate};

DROP TABLE temp.kwm_conversion purge;
CREATE TABLE if not exists temp.kwm_conversion
    (units_converted Int,
    revenue double,
    conversion_events int,
    beacon_cid int,
    beacon_rt String,
    beacon_acctid int,
    algoid int,
    beacon_dt String)
partitioned by
    (conf_wrt_dt String,
    conf_dt String)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.lazybinary.LazyBinarySerDe'
STORED AS SEQUENCEFILE;


insert overwrite table temp.kwm_conversion
partition(conf_wrt_dt, conf_dt)
select
sum(conf_units) units_converted,
sum(conf_unitprice * conf_units) revenue,
sum(1) conversion_events,
beacon_cid,
beacon_rt,
beacon_acctid,
algoid,
beacon_dt,
aat.conf_wrt_dt,
aat.conf_dt
from
    (select
     distinct
     --variables being measured
     conf_unitprice,
     conf_units,
     --group keys
     beacon_cid,
     beacon_rt,
     beacon_acctid,
     conf_wrt_dt,
     conf_dt,
     --unique identifiers
     conf_sku,
     beacon_pid,
     conf_pid
     from attribution_attributedtransactions
     where conf_wrt_dt >="${hiveconf:dts}"
     and   conf_wrt_dt <=date_add("${hiveconf:dte}",2)
     and beacon_action='click'
     and perspective='publisher'
     and clientinfo(beacon_cid,'rsx')
     and beacon_rt<>'Y'
    ) aat
join
    (select
    distinct
     if(size(split(querystring['algoid'],'%3a')) = 2, split(querystring['algoid'],'%3a')[1], querystring['algoid']) as algoid,
     cid,
     querystring['pid'] as pid,
     querystring['hmguid'] as user_id,
     dt as beacon_dt
     from beacon_logs_partitioned_daily
     where date(dt) >= date('${hiveconf:dts}')
     and date(dt) <= date('${hiveconf:dte}')
     and unix_timestamp(apache_timestamp(requesttime)) - coalesce(cast(querystring['bd'] as INT),0)  <=24*60*60*2
     and action='click'
     and clientinfo(cid,'rsx')
     and querystring['hlpt'] in ('S', 'U') and querystring['searchterms'] is not null and querystring['searchterms'] != ''
     and querystring['hmguid'] != 'fcf5d31f-7da1-46ce-b886-c4340500b066'
     and querystring['plsrc'] in ('KWS', 'Unknown')
     and cid in(131,164,193,101,162,92,180,65,299,250)
     and querystring['algoid']!=''
     and useragent not LIKE 'Mozilla/5.0 (X11% Linux x86_64% rv:10.0.10) Gecko/20121025 Firefox/10.0.10'
     and useragent not LIKE 'Mozilla/5.0 (Linux% Android 4.1% Galaxy Nexus Build/JRN84D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166 Mobile Safari/535.19 ACHEETAHI/2100050026'
     and useragent not LIKE '%Scrapy%'
     and cid!=1
)  blpd
on aat.beacon_pid=blpd.pid and aat.beacon_cid=blpd.cid
left outer join
(
   select revenue, conf_userid, conf_dt 
   from
   (
       select sum(conf_unitprice * conf_units) revenue, conf_dt, conf_userid
       from default.attribution_attributedtransactions
       where conf_wrt_dt >="${hiveconf:dts}"
       and   conf_wrt_dt <=date_add("${hiveconf:dte}",2)
       group by conf_dt, conf_userid 
   ) sus_rev
   where revenue > 5000
) suspicous_revenue
on suspicous_revenue.conf_userid = blpd.user_id and suspicous_revenue.conf_dt = aat.conf_dt
where suspicous_revenue.conf_userid is null
group by beacon_cid, beacon_rt, beacon_acctid, algoid, conf_wrt_dt, aat.conf_dt, beacon_dt
