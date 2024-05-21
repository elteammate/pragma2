'use client'

import {useEffect, useState} from "react"
import ReactCodeMirror from "@uiw/react-codemirror"
import {json} from "@codemirror/lang-json"
import {rust} from "@codemirror/lang-rust"
import {basicSetup} from "codemirror"

export default function Home() {
  const [ready, setReady] = useState(false)
  const [code, setCode] = useState("")
  const [output, setOutput] = useState({})
  const [worker, setWorker] = useState<Worker | null>(null)

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

  return (
    <main className="h-screen flex flex-row bg-black">
      <div className="h-screen w-1/2 p-3 pr-1.5">
        <ReactCodeMirror
          value={code}
          className="h-full w-full"
          height="100%"
          theme="dark"
          autoFocus={true}
          onChange={(value, _) => updateCode(value)}
          extensions={[basicSetup, rust()]}
        />
      </div>
      <div className="h-screen w-1/2 p-3 pl-1.5">
        <ReactCodeMirror
          value={ready ? JSON.stringify(output, null, 4) : "..."}
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
