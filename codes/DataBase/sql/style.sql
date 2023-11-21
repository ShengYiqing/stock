select t.* from
(
select t.*, rank() over(order by mc desc)lead_stock from
(
select t.*, rank() over(partition by l1_name, l2_name, l3_name order by mc desc) lead_ind from
(
select tl.trade_date, tl.stock_code, tl.price, tl.amount, tsb.name, ti.l1_name, ti.l2_name, ti.l3_name, ts.mc, ts.rank_mc, ts.pb, ts.rank_pb
from label.tdailylabel tl
left join tsdata.ttsstockbasic tsb
on tl.stock_code = tsb.stock_code
left join indsw.tindsw ti
on tl.stock_code = ti.stock_code
left join style.tdailystyle ts
on tl.trade_date = ts.trade_date
and tl.stock_code = ts.stock_code
where tl.trade_date = '20231116'
and tl.amount >= 30000 and tl.price >= 3
and ti.l3_name in (
#'快递', 
#'超市', '旅游零售Ⅲ', 
#'运动服装', '非运动服装', '鞋帽及其他', '其他饰品', '钟表珠宝', 
'化妆品制造及其他', '品牌化妆品', '医美服务', '医美耗材', 
'烘焙食品', '熟食', '零食', '白酒Ⅲ', '调味发酵品Ⅲ', '其他酒类', '啤酒', '保健品', '肉制品', '预加工食品', '乳品', '软饮料')
) t
) t
where t.lead_ind <= 7
) t
where t.lead_stock <= 50
order by l1_name, l2_name, l3_name, lead_stock