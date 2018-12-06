set getAclStatus=false;
set dfs.namenode.acls.enabled=false;
set mapred.reduce.tasks=5;
set hive.auto.convert.join=false;

set hive.exec.compress.output=false;

set fileName=${fileName};
set startDate=${startDate};
set endDate=${endDate};

-- 

drop table if exists temp.${hiveconf:fileName};
create external table temp.${hiveconf:fileName}
    (
    media_source_id string,
    algoid string,
    dt string,
    impressions int,
    clicks int,
    spend double,
    conversions int,
    revenue double
    )
ROW FORMAT DELIMITED
    FIELDS TERMINATED BY '\t'
stored as textfile
location  's3n://AKIAJ4CC4RABZGCAPFNA:6P2mpWbE0D3dEIydoMTvtyECWMd1ewptjxR45u4n@datascience-shared/user/hive/warehouse/DsAbTestingFinal/${hiveconf:fileName}/'
;



from
(
select
    t5.name as media_source_id,
    clickimp.algoid as algoid,
    clickimp.dt as dt,
    clickimp.impressions as impressions,
    clickimp.clicks as click,
    clickimp.spend as spend,
    COALESCE(conv.conv,0) as conversions,
    COALESCE(conv.revenue,0) as revenue
from
(
select
    t1.cid as cid,
    t1.algoid as algoid,
    t1.dt as dt,
    sum(
        if(t1.action='imp', 1, 0)
        ) as impressions,
    sum(
        if(t1.action='click', 1, 0)
        ) as clicks,
    sum(
        if(t1.action='click', t1.cost, 0)
        ) as spend
from
    (
        select
            dt,
            wrt_dt,
            cid,
            querystring['pid'] as pid,
            if(
                    size(split(querystring['algoid'],'%3a')) = 2,
                                  split(querystring['algoid'],'%3a')[1],
                                  querystring['algoid']
               ) as algoid,
            querystring['platform'] as platform,
            querystring['hlpt'] as hlpt,
            querystring['rt'] as rt,
            querystring['cost'] as cost,
            split(redirect_ip, ',')[0] as ip,
            split(split(querystring['creative'],'_')[1],'-')[0] as verticalLocation,
            split(split(querystring['creative'],'_')[1],'-')[1] as horizontalLocation,
            split(split(querystring['creative'],'_')[1],'-')[2] as grid,
            action
        from
            default.beacon_logs_partitioned_daily
        where
            date(dt) >= date('${hiveconf:startDate}')
          and date(dt) <= date('${hiveconf:endDate}')
          and querystring['hlpt'] in ('S', 'U') and querystring['searchterms'] is not null and querystring['searchterms'] != ''
          and querystring['hmguid'] != 'fcf5d31f-7da1-46ce-b886-c4340500b066'
          and action in ('imp','click') and querystring['plsrc'] in ('KWS', 'Unknown')
          and cid in(131,164,193,101,162,92,180,65,299,250)
          and querystring['algoid']!=''
          and useragent not LIKE 'Mozilla/5.0 (X11% Linux x86_64% rv:10.0.10) Gecko/20121025 Firefox/10.0.10'
          and useragent not LIKE 'Mozilla/5.0 (Linux% Android 4.1% Galaxy Nexus Build/JRN84D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166 Mobile Safari/535.19 ACHEETAHI/2100050026'
          and useragent not LIKE '%Scrapy%'
          and cid!=1
    ) t1
    left outer join
    (
        select
            distinct ip
        from
            default.ipstofilterout
    ) s
    on
        t1.ip = s.ip

  left outer join
  (
    select
        cid,
        reason,
        action,
        beacon_pid as pid
    from
        annotation.suspicious_actions
    where
        dt>='${hiveconf:startDate}'
        and dt<='${hiveconf:endDate}'
        and cid is not null
        and action in ('imp','click')
        and beacon_pid is not null
  ) sa
  on
        t1.cid=sa.cid
        and t1.pid=sa.pid
        and t1.action=sa.action
  where
        t1.algoid != -1
        and sa.pid is null
        and s.ip is null
        and t1.ip <> '10.223.242.115'
        and t1.ip <> '174.97.249.105'
group by
    t1.cid,
    t1.algoid,
    t1.dt
) clickimp
join
(
  select cid, name
  from model.hookusercompanies 
) t5
on clickimp.cid = t5.cid
left join
(
select
    sum(revenue) as revenue,
    sum(units_converted) as conv,
    algoid,
    beacon_dt as dt
from temp.kwm_conversion
where
    beacon_dt>=date('${hiveconf:startDate}')
    and beacon_dt<=date('${hiveconf:endDate}')
    and beacon_cid!=-1
group by
    beacon_dt,
    algoid
) conv
on
    clickimp.algoid = conv.algoid
    and clickimp.dt = conv.dt

where
    clickimp.impressions>=100
)tmp
insert overwrite table temp.${hiveconf:fileName}
  select *
;
