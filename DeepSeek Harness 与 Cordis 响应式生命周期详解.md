# DeepSeek Harness 与 Cordis 响应式生命周期详解

## 资料信息

- 学习材料：[ChatGPT 共享对话：学习 DeepSeek Harness](https://chatgpt.com/share/6a93b6e1-bf40-83e8-906d-b3825a8b338f)
- 核对资料：[DeepSeek Harness Architecture](https://deepseek-harness.github.io/deepseek-harness/en/reference/)
- 核对资料：[Cordis Primer](https://deepseek-harness.github.io/deepseek-harness/en/reference/cordis-primer)
- 核对资料：[Plugins and lifecycle](https://deepseek-harness.github.io/deepseek-harness/en/develop/framework/)
- 核对资料：[Services and dependencies](https://deepseek-harness.github.io/deepseek-harness/en/develop/framework/service)
- 核对资料：[Lifecycle and effects](https://deepseek-harness.github.io/deepseek-harness/en/develop/cordis-tutorial/02-lifecycle-and-effects)
- 核对资料：[Cordis API：Context](https://deepseek-harness.github.io/deepseek-harness/en/reference/cordis-api/context)
- 核对资料：[Cordis API：Fiber](https://deepseek-harness.github.io/deepseek-harness/en/reference/cordis-api/fiber)

> [!important]
> 这里的 Harness 指 DeepSeek 开源的 Agent Harness，而不是 DeepSeek 模型本身的注意力结构或推理算法。本文讨论的是它底层 Cordis 插件运行时的组合、依赖与生命周期管理。

## 一、这轮学习真正解决了什么问题

共享对话围绕一个很具体的疑问展开：插件卸载时为什么不会把已经注册的 Tool、事件监听器、定时器或 Service 留在系统里？Cordis 是否会在“安装插件”时同步记录“如何删除插件”？所谓“执行一个副作用，再加入 Tool A”究竟是什么意思？

结论可以先压缩为一句话：

> Cordis 不只是记录“如何删除整个插件”，而是在某次插件运行对应的 Fiber 中，持续收集这次运行创建的每一个可撤销 Effect；Fiber 失效或被卸载时，运行时调用这些 Effect 的 disposer，并递归清理子 Fiber。

因此，关键不是一份静态的“插件卸载脚本”，而是一个运行时维护的生命周期结构：

```text
Context 中的 Service 状态
          ↓
inject 依赖是否满足
          ↓
Fiber 应处于什么状态
          ↓
建立或撤销该 Fiber 拥有的 Effects
          ↓
Effects 又可能改变 Context 中的 Service 状态
```

这是一台动态状态机，而不是五个互不相关的 API。

## 二、先建立总图：两个相反方向的传播链

![Cordis 响应式生命周期闭环](assets/deepseek-harness/cordis-reactive-lifecycle.png)

*图：AI 生成示意图。蓝色路径表示 Provider 激活后提供 Service，满足 `inject` 并启动 Consumer；橙色路径表示 Provider 卸载后撤销 Service，使 Consumer 依赖失效并清理其 Effects。底部是 Fiber 的主要状态流转。图用于解释运行时关系，不代表源码类图。*

这张图包含两条互为镜像的传播链。

### 2.1 能力建立链

```text
Provider Fiber ACTIVE
        ↓
注册 Service 的 Effect 生效
        ↓
Service 在当前 Context 中可用
        ↓
Consumer 的 inject 全部满足
        ↓
Consumer Fiber 加载并进入 ACTIVE
        ↓
Consumer Effects 被建立
```

### 2.2 能力撤销链

```text
Provider Fiber 卸载
        ↓
Service 注册 Effect 被撤销
        ↓
Service 从当前 Context 中消失
        ↓
Consumer 的 inject 不再满足
        ↓
Consumer Fiber 卸载
        ↓
Consumer Effects 被撤销
```

如果 Consumer 自己又提供了下游 Service，这个变化还会继续向下游传播，形成级联卸载；Provider 恢复后，依赖则沿相反方向逐层恢复。

## 三、五个核心概念及其边界

### 3.1 Context：带作用域的运行环境

最初可以把 Context 理解为“插件共同生活的世界”，但这个说法不够精确。官方 API 表明，Context 是 Cordis 的核心对象，Service、事件和生命周期 API 都通过 `ctx` 到达；它还是一个代理对象，普通属性读取会进入 Service Resolver。

一个更实用的心智模型是：

```text
Context
├── Service 解析入口
├── 插件与 Fiber 注册入口
├── Effect 与事件 API
├── 当前作用域信息
└── 父子 Context / Service 隔离关系
```

`ctx.extend()` 可以创建继承父 Context 的子环境，`ctx.isolate()` 可以让某个 Service 名称进入独立作用域。因此，同名 `shell` 或 `llm` 可以在不同插件组中解析到不同实例。

需要区分两个问题：

- **Service Resolution**：当前 Context 读取 `ctx.llm` 时应该返回哪个实例？
- **Dependency Tracking**：`llm` 发生变化时，哪些 Fiber 的依赖状态需要重新计算？

两者相关，但不是同一个职责。

### 3.2 Service：具名、可替换、具有生命周期的能力

Service 不是单纯放进 Map 的对象，而是一个插件向 Context 提供的具名能力。在 Harness 中，`tools`、`llm`、`agents` 等都是 Service。

```ts
ctx.tools
ctx.llm
ctx.agents
```

Consumer 依赖的是能力名称，而不是具体 Provider 类。因此配置可以替换 LLM、Shell 或 Storage 的实现，而 Consumer 无需改动导入路径。

使用 `ctx.provide(name, value)` 注册的 Service 由当前 Fiber 拥有。官方 Context API 明确说明：当 disposer 被调用或 Provider Fiber 卸载时，这项 Service 会被注销，并唤醒依赖方重新计算。

### 3.3 inject：不是取参数，而是声明存在条件

```ts
export const inject = ['llm', 'tools']
```

这不只是说“插件代码里会用到 `llm` 和 `tools`”，而是说：只有当前 Context 中这两个必需 Service 都已准备好，这个 Fiber 才有资格加载并保持 ACTIVE。

设 Fiber 为 $F$ ，其必需 Service 集合为 $D_F$ ，Service $s$ 的可用性为 $A(s) \in \{0, 1\}$ ，则可以把这个条件写成：

$$\mathrm{Ready}(F) = \bigwedge_{s \in D_F} A(s)$$

这个公式是帮助理解的逻辑表达，不是对 Cordis 源码实现的逐字翻译。

- 若 $\mathrm{Ready}(F)=0$ ，Fiber 保持 PENDING，`apply()` 不执行。
- 若 $\mathrm{Ready}(F)$ 从 $0$ 变为 $1$ ，Fiber 可以从 PENDING 进入 LOADING。
- 若 ACTIVE 期间 $\mathrm{Ready}(F)$ 从 $1$ 变为 $0$ ，Fiber 会卸载。
- Service 恢复后，插件会再次加载，重新获取新 Service，并建立新 Effects。

因此 `inject` 不是一次性的启动前检查。它把 Consumer 的生命周期持续绑定到必需 Service 的可用性。

### 3.4 Fiber：一次插件运行的生命周期记录

Plugin 是代码定义，Fiber 是这段 Plugin 在某个 Context 中的一次运行实例。可以类比：

```text
Program  → Process
Plugin   → Fiber
```

Fiber 需要跟踪的内容包括插件定义、Context、配置、依赖状态、当前生命周期状态、注册的 Effects 以及子 Fiber。官方状态机为：

```text
PENDING → LOADING → ACTIVE → UNLOADING → DISPOSED
                    ↘ FAILED
```

| 状态 | 含义 |
| --- | --- |
| `PENDING` | 插件已经声明，但必需 Service 尚未全部准备好 |
| `LOADING` | 依赖已满足，正在执行 `apply()` 并建立资源 |
| `ACTIVE` | `apply()` 成功完成，插件处于运行状态 |
| `FAILED` | `apply()` 或配置校验抛出错误 |
| `UNLOADING` | 正在运行 disposer、撤销注册并释放资源 |
| `DISPOSED` | 此次插件运行已完全卸载 |

Fiber 不是长期不变的“插件身份”。Service 消失后旧运行会被卸载；Service 恢复时会重新加载，并建立一套新的运行状态与 Effects。

### 3.5 Effect：Acquire 与 Release 的配对

Effect 不是泛指“产生了副作用”，而是要求把外部状态的建立与撤销交给运行时管理：

```ts
ctx.effect(() => {
  const timer = setInterval(() => console.log('tick'), 1000)

  return () => {
    clearInterval(timer)
  }
})
```

可以把结构抽象成：

```text
Acquire resource
       ↓
return disposer
       ↓
Fiber stores disposer
       ↓
Fiber unloads
       ↓
run disposer
```

常见配对如下：

| Acquire | Release / disposer |
| --- | --- |
| `setInterval()` | `clearInterval()` |
| `addListener()` | `removeListener()` |
| `openConnection()` | `closeConnection()` |
| `registerTool()` | `unregisterTool()` |
| `ctx.plugin(child)` | `child.dispose()` |
| `ctx.provide('llm', value)` | 注销 `llm` Service |

Cordis 已经管理的注册通常本身就是 Effect，例如 Service 注册、`ctx.on()`、`ctx.plugin(child)`，以及 Harness 的 Tool 注册。只有 Cordis 不认识的外部资源，才需要插件作者显式包进 `ctx.effect()`。

## 四、直接回答“副作用 + Tool A”是什么意思

假设插件执行：

```ts
export const inject = ['llm']

export function apply(ctx: Context) {
  ctx.tools.register(toolA)
}
```

“执行一个副作用 + Tool A”真正表达的是：

1. `apply()` 在外部 Tool Registry 中加入了 `toolA`。
2. 这个注册动作不是只返回一个永久存在的对象，而是有对应的撤销动作。
3. Cordis/Harness 把该注册的 disposer 归属于当前 Fiber。
4. 当 `llm` 消失使该 Fiber 卸载，或插件被配置移除、热更新、显式 `dispose()` 时，运行时会撤销 `toolA`。

概念上等价于：

```ts
ctx.effect(() => {
  const unregister = registerTool(toolA)
  return () => unregister()
})
```

但如果 `ctx.tools.register()` 本身已经是 Cordis 管理的注册 API，就不应再机械地套一层重复 Effect；调用该 API 时，返回的 disposer 已经会附着到当前插件的 Fiber。

所以，Cordis 记录的不是“Tool A 的删除说明文档”，而是这次真实注册所返回的可执行撤销函数，以及它属于哪个 Fiber。

## 五、为什么 Effect 通常逆序清理

假设加载顺序是：

```text
1. 打开 Database
2. 基于 Database 创建 Repository
3. 基于 Repository 启动 Worker
```

后建立的资源依赖先建立的资源，因此卸载必须按相反方向进行：

```text
Worker → Repository → Database
```

这就是 LIFO：后注册的 disposer 先启动。官方文档确认，Fiber 卸载时 disposer 按注册顺序的逆序启动。

但要注意一个重要边界：多个异步 disposer 可能并发运行。“逆序启动”不等于“前一个完全结束后才启动下一个”。如果清理步骤必须严格串行，应把它们放进同一个 disposer：

```ts
ctx.effect(() => {
  return async () => {
    await stopWorker()
    await closeDatabase()
  }
})
```

## 六、加载失败为何不会留下半个插件

假设 `apply()` 的执行过程是：

```text
注册 Tool ✓
启动 Timer ✓
打开 Connection ✗
```

如果失败后不清理，模型仍可能看到已经注册的 Tool，但真正执行 Tool 所需的连接不存在。这种 partial initialization 比直接启动失败更危险。

因此，Fiber 在 LOADING 期间也收集已经成功建立的 Effects。后续步骤失败时，需要撤销此前已经建立的资源，再进入 FAILED。可以用“生命周期事务”帮助理解：

```text
BEGIN LOAD
  Effect A ✓
  Effect B ✓
  Effect C ✗
ROLLBACK
  Dispose B
  Dispose A
```

“事务”是解释模型，不意味着 Cordis 提供数据库式 ACID 保证；它强调的是失败时不应遗留半安装状态。

## 七、时间组合性到底是什么

最初把时间组合性理解成“插件可以干净卸载”并没有错，但范围太窄。更准确的理解是：

> 组件可以在任意时刻进入或离开系统，运行时保证它在存活区间中建立的外部状态与其生命周期保持一致。

设 Fiber 的存活区间为 $L_F$ ，某个 Effect 的存活区间为 $L_E$ ，则应满足：

$$L_E \subseteq L_F$$

默认情况下，Effect 随 Fiber 建立并随 Fiber 撤销，可以近似理解为：

$$L_E = L_F$$

如果插件提前调用 `ctx.effect()` 返回的 disposer，则 Effect 可以比 Fiber 更早结束：

$$L_E \subset L_F$$

`ctx.effect()` 返回的 disposer 支持提前释放；官方 API 还明确说明，重复调用该 disposer 是 no-op。这避免了“手动提前释放一次，Fiber 卸载时又释放一次”导致的重复关闭问题。

空间组合性则关注“组件在哪个 Context/Scope 中可见、同名 Service 如何隔离”。两者合起来，Cordis 才能支持：

- 不同 Agent Context 使用不同 LLM 或 Shell 实例；
- Tool、Skill、Subagent、Session 在局部作用域内创建与销毁；
- Provider 热替换时，只重启受到依赖影响的消费者；
- Runtime 持续运行，而局部能力动态出现、消失、替换和重组。

## 八、父子 Fiber 与结构化生命周期

`ctx.plugin(childPlugin)` 创建的是一个独立的子 Fiber，但它的生命周期归属于父 Fiber：

```text
Agent Fiber
├── Planner Fiber
├── Memory Fiber
└── Subagent Fiber
```

父 Fiber 卸载时，子 Fiber 会被递归卸载，异步 cleanup 完成后 `fiber.dispose()` 才结束。这个结构防止父组件已经消失而子插件仍留在系统里，成为孤儿组件。

其原则类似结构化并发：子任务可以有独立状态，但不应无意中活得比父作用域更久。

## 九、必需依赖与可选依赖不要混用

必需依赖：没有该 Service，插件整体就不应该运行。

```ts
export const inject = ['tools']
```

可选依赖：没有该 Service，插件仍能完成主要功能。

```ts
export function apply(ctx: Context) {
  const metrics = ctx.get('metrics')
  metrics?.record('plugin_loaded', 1)
}
```

如果把可选的 `metrics` 错写进 `inject`，那么 `metrics` Provider 一旦消失，整个插件都会卸载。这不是普通的空值处理差异，而是生命周期语义差异。

## 十、Service 替换为什么需要重跑消费者

假设 Consumer 在 `apply()` 中捕获了旧实例：

```ts
const llm = ctx.llm

ctx.on('message', async (message) => {
  await llm.generate(message)
})
```

即使 Context 后来能解析到新 Provider，这个闭包里的 `llm` 仍指向旧实例。如果不重启 Consumer，就可能出现一部分代码使用旧 Service、另一部分使用新 Service 的撕裂状态。

Cordis 的策略更彻底：

```text
required Service changed
        ↓
Consumer unload
        ↓
撤销旧 Effects 和旧闭包
        ↓
重新 apply
        ↓
获取新 Service 并建立新 Effects
```

HMR 由此不需要为每种插件单独发明热切换协议：旧 Provider 卸载、注册清理、新 Provider 挂载、依赖消费者重新加载，都沿用同一套生命周期机制。

## 十一、Mini-Cordis：用代码还原核心闭环

下面的代码只演示 `Service → inject → Fiber → Effect` 主线。它不是 Cordis 的源码替代品，也没有实现 Context Scope、Proxy、父子 Fiber、异步 Effect、错误聚合或防重入调度。

```ts
type Disposer = () => void

type Plugin = {
  name: string
  inject?: string[]
  apply: (ctx: PluginContext) => void
}

type PluginContext = {
  provide(name: string, value: unknown): Disposer
  get<T>(name: string): T | undefined
  has(name: string): boolean
  effect(execute: () => void | Disposer): Disposer
}

class Context {
  private services = new Map<string, unknown>()
  private fibers = new Set<Fiber>()
  private refreshing = false
  private refreshRequested = false

  provide(name: string, value: unknown): Disposer {
    this.services.set(name, value)
    this.requestRefresh()

    let disposed = false
    return () => {
      if (disposed) return
      disposed = true
      this.services.delete(name)
      this.requestRefresh()
    }
  }

  get<T>(name: string): T | undefined {
    return this.services.get(name) as T | undefined
  }

  has(name: string): boolean {
    return this.services.has(name)
  }

  plugin(plugin: Plugin): Fiber {
    const fiber = new Fiber(this, plugin)
    this.fibers.add(fiber)
    this.requestRefresh()
    return fiber
  }

  removeFiber(fiber: Fiber): void {
    this.fibers.delete(fiber)
  }

  // 教学版：批量刷新，避免在插件启动期间递归进入刷新逻辑。
  private requestRefresh(): void {
    this.refreshRequested = true
    if (this.refreshing) return

    this.refreshing = true
    try {
      while (this.refreshRequested) {
        this.refreshRequested = false
        for (const fiber of [...this.fibers]) fiber.refresh()
      }
    } finally {
      this.refreshing = false
    }
  }
}

class Fiber {
  private active = false
  private disposed = false
  private disposers: Disposer[] = []

  constructor(
    private readonly ctx: Context,
    private readonly plugin: Plugin,
  ) {}

  refresh(): void {
    if (this.disposed) return

    const ready = (this.plugin.inject ?? [])
      .every((name) => this.ctx.has(name))

    if (!this.active && ready) this.start()
    if (this.active && !ready) this.stop()
  }

  dispose(): void {
    if (this.disposed) return
    this.stop()
    this.disposed = true
    this.ctx.removeFiber(this)
  }

  private start(): void {
    this.active = true
    try {
      this.plugin.apply(this.createPluginContext())
    } catch (error) {
      this.cleanupEffects()
      this.active = false
      throw error
    }
  }

  private stop(): void {
    if (!this.active) return
    this.cleanupEffects()
    this.active = false
  }

  private createPluginContext(): PluginContext {
    const fiber = this
    return {
      provide: this.ctx.provide.bind(this.ctx),
      get: this.ctx.get.bind(this.ctx),
      has: this.ctx.has.bind(this.ctx),
      effect(execute): Disposer {
        const cleanup = execute() ?? (() => {})
        let disposed = false

        const dispose = () => {
          if (disposed) return
          disposed = true
          const index = fiber.disposers.indexOf(dispose)
          if (index >= 0) fiber.disposers.splice(index, 1)
          cleanup()
        }

        fiber.disposers.push(dispose)
        return dispose
      },
    }
  }

  private cleanupEffects(): void {
    for (const dispose of [...this.disposers].reverse()) dispose()
    this.disposers = []
  }
}
```

示例 Provider：

```ts
const llmPlugin: Plugin = {
  name: 'llm-provider',
  apply(ctx) {
    const llm = {
      generate: (prompt: string) => `DeepSeek response to: ${prompt}`,
    }

    ctx.effect(() => ctx.provide('llm', llm))
  },
}
```

示例 Consumer：

```ts
const agentPlugin: Plugin = {
  name: 'agent',
  inject: ['llm'],
  apply(ctx) {
    const llm = ctx.get<{ generate(prompt: string): string }>('llm')!
    console.log(llm.generate('hello'))

    ctx.effect(() => {
      const timer = setInterval(() => console.log('heartbeat'), 1000)
      return () => clearInterval(timer)
    })
  },
}

const ctx = new Context()
ctx.plugin(agentPlugin)       // llm 不存在，Agent 保持 PENDING
const llmFiber = ctx.plugin(llmPlugin)
// llm 出现，Agent 自动启动

llmFiber.dispose()
// llm 消失，Agent 自动停止，heartbeat 被清理
```

### 11.1 这份 Mini-Cordis 展示了什么

- `provide()` 改变 Service 可用性，并触发依赖刷新。
- `inject` 被落实为 Fiber 是否应该 ACTIVE 的谓词。
- `effect()` 立即建立资源，同时把幂等 disposer 记录进当前 Fiber。
- Fiber 停止时按逆序清理 Effects。
- Provider Service 消失会驱动 Consumer 停止。

### 11.2 它没有展示什么

- 每次刷新仍扫描全部 Fiber，复杂度近似为 $O(N)$ ，其中 $N$ 是 Fiber 数量。
- 没有真实 Cordis 的 Service 作用域、隔离与 Proxy 解析。
- 没有父子 Fiber 树和递归异步清理。
- 没有 Promise、异步生成器 Effect、错误聚合与诊断标签。
- 批量刷新只是教学性的防重入处理，不能据此推断真实 Cordis 使用同样的队列结构。

真实运行时可以通过依赖索引减少无关 Fiber 的重算，这是合理的工程方向；但在没有直接查看对应源码前，不应断言它内部一定存在名为 `DependencyIndex` 的 Map，或一定使用某种 microtask/transaction 调度器。

> [!note]
> 本次已对示例的类型关系、Context 方法绑定、依赖刷新、幂等释放和逆序清理进行静态一致性检查；当前环境没有 TypeScript 编译器，因此没有把它标记为已编译或已运行验证。它的定位仍是公式与生命周期主线一致的教学代码。

## 十二、常见误解与纠正

### 12.1 “安装插件时记录一个统一的卸载函数”

不够准确。Fiber 在插件加载过程中持续收集每个 Effect 的 disposer；内置注册 API 也会把撤销逻辑绑定到当前 Fiber。它更像动态增长的 cleanup stack。

### 12.2 “inject 就是依赖注入参数”

不够准确。它还定义 Fiber 的生命周期前提，并在加载之后持续跟踪 Service 变化。

### 12.3 “Service 消失只是把 ctx.llm 设为 undefined”

错误。对于把 `llm` 声明为必需依赖的 Consumer，Service 消失会驱动 Consumer 卸载，防止旧闭包继续持有失效实例。

### 12.4 “逆序清理意味着异步 disposer 严格串行”

错误。官方只保证逆注册顺序启动；需要严格顺序时，应在同一个 disposer 中显式 `await`。

### 12.5 “Mini-Cordis 就是真实 Cordis 的内部源码”

错误。Mini 版本是用于验证心智模型的教学实现。真实 Cordis 还处理作用域、隔离、父子 Fiber、异步 Effect、错误、诊断与热更新等问题。

## 十三、为什么这套机制特别适合 Agent Harness

Agent Harness 的运行环境比普通“启动后一直运行”的服务更动态：

- 运行中切换模型 Provider；
- 动态挂载或撤销 Tool 与 Skill；
- 创建和销毁 Subagent；
- 为不同 Session 建立局部运行环境；
- 替换 Sandbox、Shell 或 Storage；
- 热更新插件与配置。

如果每种能力都自己实现一套 `start/stop/reconnect/reload` 协议，组件之间会形成大量隐式耦合。Cordis 把共同问题统一为：

```text
能力是否存在？       → Service
组件何时可以运行？   → inject
这次运行由谁管理？   → Fiber
创建了哪些外部状态？ → Effect
这些状态在哪个范围？ → Context / Scope
```

因此可以把 Cordis 概括为：

> 一个作用域化、依赖驱动、支持可逆 Effect 的插件生命周期 Runtime。

## 十四、对共享对话内容的证据分层

### 官方文档直接确认

- Harness 以 Cordis 为底层插件框架，模型适配器、Tool Registry、Session Log、Agent Loop 等都可作为插件组合。
- Fiber 具有 PENDING、LOADING、ACTIVE、FAILED、UNLOADING、DISPOSED 状态。
- 必需 Service 不可用时插件保持 PENDING；Service 消失时依赖插件自动卸载，恢复后重新加载。
- Service 注册、事件监听、子插件和 Harness Registry 注册属于可逆 Effect。
- disposer 逆注册顺序启动；异步 disposer 可能并发。
- `ctx.plugin(child)` 创建子 Fiber，父 Fiber 卸载时递归清理。
- `ctx.effect()` 返回的 disposer 可提前调用，重复调用是 no-op。
- 可选依赖应在使用点通过 `ctx.get()` 探测，而不是写入 `inject`。

### 有助理解，但属于解释模型

- 把 Fiber 类比为进程或“生命周期事务单元”。
- 用 $\mathrm{Ready}(F)$ 表示 Fiber 的 ACTIVE 条件。
- 把 Effects 看成 cleanup stack。
- 用能力传播链与卸载传播链解释响应式闭环。

### 需要源码证据才能确认的实现细节

- 是否存在某个具体命名和数据结构的 Service-to-Fiber 依赖索引。
- 真实运行时是否通过 microtask、transaction 或 batch 处理所有重入场景。
- 某个版本中 Provider 替换、失败回滚和并发清理的精确调度顺序。

DeepSeek Harness 当前仍处于快速迭代阶段，API 与内部实现可能发生不兼容变化。后续若进入源码学习，应以当前 checkout 的 `vendor/cordis/` 与相应版本文档为准。

## 十五、复习时只记住这条主线

```text
Context 决定能力在哪个作用域中可见
        ↓
Service 表示可动态出现、消失和替换的能力
        ↓
inject 把必需能力变成 Fiber 的运行条件
        ↓
Fiber 管理一次插件运行的状态、子插件和 Effects
        ↓
Effect 把 Acquire 与 Release 配对
        ↓
Service 变化再次驱动 inject 与 Fiber
```

最终，空间组合性回答“组件放在哪里、能看到什么”，时间组合性回答“组件在何时存在、离开时留下的状态能否一起消失”。Cordis 用 Context、Service、inject、Fiber 与 Effect 把这两个问题统一进同一个运行时模型。
