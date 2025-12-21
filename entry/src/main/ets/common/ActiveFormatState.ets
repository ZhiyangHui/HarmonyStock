@ObservedV2
export class ActiveFormState {
  // 当前被激活的桌面卡片 FormId，用于从桌面卡片跳转回应用
  @Trace formId: string = ''
}

// AppStorage 中用于保存 ActiveFormState 的 key
export const APP_STORAGE_KEY_ACTIVE_FORM: string = 'active_form_state_v2'

// =============================
// 通知点击跳转相关的独立状态
// =============================
@ObservedV2
export class ActiveQuoteNavState {
  @Trace targetCode: string = ''            // 通知希望跳转到的股票或指数代码
  @Trace targetType: string = ''            // 目标类型，用于区分 stock 还是 index
  @Trace klinePeriod: string = ''           // 跳转后默认展示的 K 线周期
  @Trace ensureInWatchlist: boolean = true  // 如果不在自选列表中，是否需要自动加入

  // 每次通知跳转时递增，用来触发页面侧重新处理跳转逻辑
  @Trace version: number = 0
}

// AppStorage 中用于保存通知跳转状态的 key
export const APP_STORAGE_KEY_QUOTE_NAV: string = 'active_quote_nav_state_v2'

// router / want 中用于传递桌面卡片 FormId 的参数 key
export const ROUTER_KEY_FORM_ID: string = 'formId'

// 通知跳转时通过 router / want 传递的参数 key
export const ROUTER_KEY_TARGET_CODE: string = 'targetCode'
export const ROUTER_KEY_TARGET_TYPE: string = 'targetType'
export const ROUTER_KEY_KLINE_PERIOD: string = 'klinePeriod'
export const ROUTER_KEY_ENSURE_WATCHLIST: string = 'ensureInWatchlist'
