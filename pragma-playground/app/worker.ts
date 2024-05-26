// This is a module worker, so we can use imports (in the browser too!)
import init, {compile} from "pragma-wasm"
import JSCPP from "JSCPP"

let ready = false

const do_compile = (code: string) => {
  console.log(code)
  try {
    let compiled = compile(code) as any
    if (compiled.Ok) {
      let c = compiled.Ok.c
      let out = ""
      try {
        const exitcode = JSCPP.run(c, "", {
          stdio: {
            write: (s: string) => out += s,
          },
          maxTimeout: 1000,
        })
        out += `\n\nFinished with exit code: ${exitcode}\nWarning: JSCPP is not a full C++ compiler, so some code may not work as expected.`
      } catch (e: any) {
        out += e.toString()
      }

      return {
        Ok: {
          out,
          ...compiled.Ok,
        }
      }
    }
    return compiled
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
