(function(){
	function parseSpec(el){
		var id=el.getAttribute("data-apg-chart");
		var script=id?document.getElementById(id):el.querySelector('script[type="application/json"]');
		if(!script)return null;
		try{return JSON.parse(script.textContent||"{}");}catch(e){return null;}
	}
	function empty(el,msg){
		el.innerHTML='<div class="apg-chart-empty"><p>'+escapeHtml(msg||"No data available")+'</p></div>';
	}
	function escapeHtml(s){return String(s).replace(/[&<>"']/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c];});}
	function svgChart(el,spec){
		var data=Array.isArray(spec.data)?spec.data:[];
		if(!data.length){empty(el,spec.empty);return;}
		var total=data.reduce(function(n,d){return n+Number(d.value||0);},0);
		if(total<=0){empty(el,spec.empty);return;}
		if(spec.type==="progress"){
			var value=Math.max(0,Math.min(100,Number(data[0].value||0)));
			el.innerHTML='<div class="apg-progress"><span style="width:'+value+'%"></span></div><p class="text-sm text-gray-500 mt-2">'+value+'%</p>'+table(data);
			return;
		}
		var offset=25, circles='';
		data.forEach(function(d,i){
			var pct=Number(d.value||0)/total*100;
			circles+='<circle r="15.9" cx="18" cy="18" fill="transparent" stroke="var(--apg-chart-'+(i%6)+')" stroke-width="7" stroke-dasharray="'+pct+' '+(100-pct)+'" stroke-dashoffset="-'+offset+'"></circle>';
			offset+=pct;
		});
		el.innerHTML='<svg class="apg-donut" viewBox="0 0 36 36" role="img" aria-label="'+escapeHtml(spec.title||"Chart")+'">'+circles+'</svg>'+table(data);
	}
	function uplotChart(el,spec){
		var rows=Array.isArray(spec.data)?spec.data:[];
		if(!rows.length||!window.uPlot){empty(el,spec.empty);return;}
		var x=rows.map(function(r,i){return Number(r.x==null?i:r.x);});
		var y=rows.map(function(r){return Number(r.y||r.value||0);});
		var opts={width:Math.max(280,el.clientWidth||360),height:Number(spec.height||180),series:[{}, {label:spec.title||"Value",stroke:getComputedStyle(document.documentElement).getPropertyValue("--apg-primary")||"#1E5B5A",fill:spec.type==="area"?"rgba(30,91,90,.12)":undefined}],axes:[{},{}]};
		el.innerHTML="";
		new uPlot(opts,[x,y],el);
		el.insertAdjacentHTML("beforeend",table(rows));
	}
	function table(rows){
		return '<details class="apg-chart-data"><summary>Data table</summary><table class="apg-table"><tbody>'+rows.map(function(r){return '<tr><td>'+escapeHtml(r.label||r.x||"")+'</td><td>'+escapeHtml(r.value==null?r.y:r.value)+'</td></tr>';}).join("")+'</tbody></table></details>';
	}
	function hydrate(){
		document.querySelectorAll("[data-apg-chart]").forEach(function(el){
			var spec=parseSpec(el);
			if(!spec){empty(el,"Chart data unavailable");return;}
			if(["donut","progress"].indexOf(spec.type)>=0)svgChart(el,spec);else uplotChart(el,spec);
		});
	}
	document.addEventListener("DOMContentLoaded",hydrate);
	window.addEventListener("apg:themechange",hydrate);
})();
