// This is a module worker, so we can use imports (in the browser too!)
import init, {compile} from "pragma-wasm"

let ready = false

const do_compile = (code: string) => {
  console.log(code)
  try {
    return compile(code)
  } catch (e: any) {
    return {error: e.toString()}
  }
}

addEventListener("message", (event: MessageEvent<string>) => {
  if (!ready) {
    init().then(() => {
      ready = true
      postMessage(do_compile(event.data))
    })
  } else {
    postMessage(do_compile(event.data))
  }
});
