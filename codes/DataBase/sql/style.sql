select * from
( 
select t.* from
(
select t.*, rank() over(order by mc desc)lead_stock from
(
select t.*, rank() over(partition by l1_name, l2_name, l3_name order by mc desc) lead_ind from
(
select tl.trade_date, tl.stock_code, tl.price, tl.amount, tsb.name, ti.l1_name, ti.l2_name, ti.l3_name, ts.beta, ts.rank_beta, ts.mc, ts.rank_mc, ts.pb, ts.rank_pb
from label.tdailylabel tl
left join tsdata.ttsstockbasic tsb
on tl.stock_code = tsb.stock_code
left join indsw.tindsw ti
on tl.stock_code = ti.stock_code
left join style.tdailystyle ts
on tl.trade_date = ts.trade_date
and tl.stock_code = ts.stock_code
where tl.trade_date = '20231215'
and tl.amount >= 30000 and tl.price >= 3
and ts.rank_mc >= 0.2
) t
) t
where t.lead_ind <= 500
) t
where t.lead_stock <= 3800
order by l1_name, l2_name, l3_name, lead_stock
) t