(function(){
	var retries=0, source=null, bindings=[];
	function escapeHtml(s){return String(s).replace(/[&<>"']/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c];});}
	function topics(){
		var set={};
		document.querySelectorAll("[data-apg-live]").forEach(function(el){
			String(el.getAttribute("data-apg-live")||"").split(",").forEach(function(t){t=t.trim();if(t)set[t]=true;});
		});
		return Object.keys(set);
	}
	function pill(el,count){
		var badge=el.querySelector("[data-apg-live-count]");
		if(!badge){
			badge=document.createElement("button");
			badge.type="button";
			badge.className="apg-live-pill";
			badge.setAttribute("data-apg-live-count","0");
			badge.addEventListener("click",function(){location.reload();});
			el.prepend(badge);
		}
		badge.textContent=count+" new "+(count===1?"record":"records")+" - refresh";
		badge.hidden=false;
	}
	function handle(event){
		var payload={};
		try{payload=JSON.parse(event.data||"{}");}catch(e){return;}
		document.querySelectorAll("[data-apg-live]").forEach(function(el){
			var live=String(el.getAttribute("data-apg-live")||"");
			if(live.split(",").map(function(t){return t.trim();}).indexOf(payload.topic)<0)return;
			var count=Number(el.getAttribute("data-apg-live-new")||"0")+1;
			el.setAttribute("data-apg-live-new",String(count));
			pill(el,count);
			el.dispatchEvent(new CustomEvent("apg:live",{detail:payload,bubbles:true}));
		});
	}
	function connect(){
		var ts=topics();
		if(!ts.length||!window.EventSource)return;
		if(source)source.close();
		source=new EventSource("/events?topics="+encodeURIComponent(ts.join(",")));
		source.addEventListener("apg-ready",function(){retries=0;});
		source.addEventListener("record",handle);
		source.addEventListener("workflow",handle);
		source.addEventListener("agent-token",handle);
		source.addEventListener("agent-result",handle);
		source.onerror=function(){
			if(source)source.close();
			var delay=Math.min(30000,1000*Math.pow(2,retries++));
			setTimeout(connect,delay);
		};
	}
	function renderMarkdown(text){
		var safe=escapeHtml(text);
		safe=safe.replace(/`([^`]+)`/g,"<code>$1</code>");
		safe=safe.replace(/\*\*([^*]+)\*\*/g,"<strong>$1</strong>");
		safe=safe.replace(/(?:^|\n)- ([^\n]+)/g,function(_,item){return "<ul><li>"+item+"</li></ul>";});
		return safe.replace(/\n/g,"<br>");
	}
	function bindConsole(form){
		form.addEventListener("submit",function(){
			var output=document.querySelector(form.getAttribute("data-apg-stream-target")||"");
			if(output)output.innerHTML='<span class="apg-stream-cursor">Streaming...</span>';
		});
	}
	document.addEventListener("DOMContentLoaded",function(){
		bindings=Array.prototype.slice.call(document.querySelectorAll("[data-apg-live]"));
		document.querySelectorAll("form[data-apg-stream-target]").forEach(bindConsole);
		connect();
	});
	window.apgRenderMarkdown=renderMarkdown;
})();
