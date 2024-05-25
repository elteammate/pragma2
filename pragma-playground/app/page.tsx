'use client'

import {ReactNode, useEffect, useMemo, useRef, useState} from "react"
import ReactCodeMirror, {
  Decoration,
  EditorView,
  hoverTooltip, ReactCodeMirrorRef, StateEffect, StateField,
} from "@uiw/react-codemirror"
import {json} from "@codemirror/lang-json"
import {rust} from "@codemirror/lang-rust"
import {createRoot} from "react-dom/client"

type Message = {
  from: number,
  to: number,
  message: string,
}

const addError = StateEffect.define<Message>()

const errorMark = Decoration.mark({class: "cm-error"})

const errorsField = StateField.define({
  create() {
    return Decoration.none
  },
  update(underlines, tr) {
    underlines = underlines.map(tr.changes)
    for (let e of tr.effects) if (e.is(addError)) {
      underlines = underlines.update({
        add: [errorMark.range(e.value.from, e.value.to)]
      })
    }
    return underlines
  },
  provide: f => EditorView.decorations.from(f)
})

const messagesField: StateField<Message[]> = StateField.define({
  create() {
    return []
  },
  update(messages: Message[], tr) {
    messages = [...messages]
    for (let e of tr.effects) if (e.is(addError)) {
      messages.push(e.value)
    }
    return messages
  }
})

const errorHover = hoverTooltip((view, pos) => {
  const errors: ReactNode[] = []

  const messages = view.state.field(messagesField, false) ?? []

  for (let i = 0; i < messages.length; i++){
    const m = messages[i]
    if (m.from <= pos && pos < m.to) {
      let text = m.message
      let chunks = text.split("`")
      let parts: ReactNode[] = []
      for (let i = 0; i < chunks.length; i++) {
        if (i % 2 == 0) {
          parts.push(<p key={i}>{chunks[i]}</p>)
        } else {
          parts.push(<pre key={i} className="text-amber-200">{chunks[i]}</pre>)
        }
      }
      errors.push(<div className="p-2 m-1 rounded-md bg-gray-700">{parts}</div>)
    }
  }

  const element = document.createElement("div")
  const root = createRoot(element)
  if (errors.length != 0) {
    root.render(<div className="w-[500px] p-2 rounded-md overflow-y-auto max-h-96 text-xs">
      {errors}
    </div>)
  }

  return {
    pos: pos - 1,
    end: pos + 100,
    above: true,
    create() {
      return {dom: element}
    }
  }
}, {
  hoverTime: 1,
  hideOnChange: true,
})

export default function Home() {
  const [ready, setReady] = useState(false)
  const [code, setCode] = useState(() => {
    if (typeof localStorage === "undefined") return ""
    const code = localStorage.getItem("code") ?? ""
    if (code) return code
    return ""
  })
  const [output, setOutput] = useState({})
  const [worker, setWorker] = useState<Worker | null>(null)
  // const [errorHover, setErrorHover] = useState<Extension>(() => createErrorHover([]))

  const createWorker = () => {
    const worker = new Worker(new URL("./worker.ts", import.meta.url))
    worker.onmessage = (event) => {
      setReady(true)
      setOutput(event.data)
    }
    setWorker(worker)
    setReady(true)
  }

  const updateCode = (value: string) => {
    setCode(value)
    if (ready) {
      worker!.postMessage(value)
      setReady(false)
    } else {
      worker?.terminate()
      createWorker()
    }
  }

  useEffect(() => {
    createWorker()
    updateCode(code)
    return () => {
      worker?.terminate()
    }
  }, [])

  useEffect(() => {
    if (worker)
      updateCode(code)
  }, [worker])

  useEffect(() => {
    if (localStorage)
      localStorage.setItem("code", code)
  }, [code])

  const refs = useRef<ReactCodeMirrorRef>({})

  const underlineTheme = useMemo(() => EditorView.baseTheme({
    ".cm-error": {
      textDecoration: "underline 2px red",
    }
  }), [])

  const out = output as any
  let messages = []
  if (out.Err) {
    if (out.Err.ParseError) {
      for (const err of out.Err.ParseError) {
        if (err.Spanned) {
          const [[begin, end], msg] = err.Spanned
          messages.push({
            from: begin,
            to: end,
            message: msg,
          })
        } else if (err.UnexpectedEof) {
          const msg = err.UnexpectedEof
          messages.push({
            from: code.length - 1,
            to: code.length,
            message: msg,
          })
        }
      }
    } else if (out.Err.ElaborateError) {
      for (const err of out.Err.ElaborateError) {
        const [[begin, end], msg] = Object.values(err)[0] as any
        messages.push({
          from: begin,
          to: end,
          message: msg,
        })
      }
    }
  }

  useEffect(() => {
    if (!refs.current || !refs.current.view) return
    const view = refs.current.view

    let effects: StateEffect<any>[] = messages
      .filter((m) => m.from !== m.to)
      .map((m) => addError.of(m))

    if (!effects.length) return

    if (!view.state.field(errorsField, false))
      effects.push(StateEffect.appendConfig.of([errorsField, messagesField, underlineTheme]))

    view.dispatch({effects})
  }, [refs.current, output])

  let result
  if (!ready) {
    result = "..."
  } else if (out.Ok) {
    result = out.Ok.ctx;
  } else {
    result = JSON.stringify(output, (key, value) => {
      if (key === "span") return undefined
      if (typeof value === "bigint")
        return Number(value)
      return value
    }, 2)
  }

  return (
    <main className="h-screen flex flex-row bg-black">
      <div className="h-screen w-1/2 p-3 pr-1.5">
        <ReactCodeMirror
          value={code}
          ref={refs}
          className="h-full w-full"
          height="100%"
          theme="dark"
          autoFocus={true}
          onChange={(value, _) => updateCode(value)}
          extensions={[rust(), errorHover]}
        />
      </div>
      <div className="h-screen w-1/2 p-3 pl-1.5">
        <ReactCodeMirror
          value={result}
          editable={false}
          className="h-full w-full"
          height="100%"
          width="100%"
          theme="dark"
          extensions={[json()]}
        />
      </div>
    </main>
  );
}
