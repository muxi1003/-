const {chromium}=require('C:/Users/muxi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
 const b=await chromium.launch({channel:'msedge',headless:true});
 const p=await b.newPage();
 await p.setContent('<input type="file"><video controls></video>');
 await p.evaluate(()=>{document.querySelector('input').onchange=e=>document.querySelector('video').src=URL.createObjectURL(e.target.files[0]);});
 await p.locator('input').setInputFiles('E:/real/use_code/yoloV8/Experiment/03_可靠事件参考/event_workspace/browser_media_v1/16170075_browser_lossless.mp4');
 await p.waitForTimeout(3000);
 const result=await p.locator('video').evaluate(v=>({ready:v.readyState,duration:v.duration,error:v.error?.message}));
 console.log(result);await b.close();
 if(result.ready<1)process.exit(1);
})().catch(e=>{console.error(e);process.exit(1)});
