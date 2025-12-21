import { AbilityConstant, ConfigurationConstant, UIAbility, Want } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { window, AppStorageV2 } from '@kit.ArkUI';
import {
  ActiveFormState,
  APP_STORAGE_KEY_ACTIVE_FORM,
  ROUTER_KEY_FORM_ID,

  // 通知点击跳转相关的全局状态与 AppStorage key
  ActiveQuoteNavState,
  APP_STORAGE_KEY_QUOTE_NAV,
} from '../common/ActiveFormatState'

// hilog 日志域，通常保持固定即可
const DOMAIN = 0x0000;

// 当前 Ability 使用的日志 TAG
const TAG: string = 'HarmonyStock';

// 从通知 want.parameters 中解析跳转参数用的临时结构
class QuoteNavParams {
  targetCode: string = ''              // 目标股票或指数代码
  targetType: string = ''              // 目标类型：stock / index
  klinePeriod: string = ''             // 默认展示的 K 线周期
  ensureInWatchlist: boolean = true    // 是否确保加入自选
}

export default class EntryAbility extends UIAbility {
  // Ability 创建时触发，冷启动或被系统重新拉起都会走到这里
  onCreate(want: Want, launchParam: AbilityConstant.LaunchParam): void {
    try {
      // 禁止跟随系统深浅色模式，主题完全由应用自身控制
      this.context
        .getApplicationContext()
        .setColorMode(ConfigurationConstant.ColorMode.COLOR_MODE_NOT_SET)
    } catch (err) {
      // 主题模式设置失败不影响主流程，仅记录日志
      hilog.error(
        DOMAIN,
        TAG,
        'Failed to set colorMode. Cause: %{public}s',
        JSON.stringify(err)
      )
    }

    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onCreate')
    this.dumpWantKeys(want, 'onCreat')

    // 冷启动时统一处理桌面卡片（Form）跳转参数
    this.consumeFormIdIntoGlobalState(want, 'onCreate')

    // 冷启动时统一处理通知点击带来的行情跳转参数
    this.consumeQuoteNavIntoGlobalState(want, 'onCreate')

    // 打印完整 parameters，方便排查通知或 router 传参问题
    hilog.info(DOMAIN, TAG, 'want.parameters=%{public}s', JSON.stringify(want.parameters))
  }

  // Ability 已存在时，再次被拉起会走这里（如通知点击）
  onNewWant(want: Want, launchParam: AbilityConstant.LaunchParam): void {
    this.dumpWantKeys(want, 'onNewWant')
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onNewWant')

    // 处理可能来自桌面卡片的 formId
    this.consumeFormIdIntoGlobalState(want, 'onNewWant')

    // 处理可能来自通知的行情跳转参数
    this.consumeQuoteNavIntoGlobalState(want, 'onNewWant')

    hilog.info(DOMAIN, TAG, '[onNewWant] want.parameters=%{public}s', JSON.stringify(want.parameters))
  }

  // Ability 销毁时调用
  onDestroy(): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onDestroy');
  }

  // WindowStage 创建时加载页面入口
  onWindowStageCreate(windowStage: window.WindowStage): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onWindowStageCreate');

    // 加载主页面 Index
    windowStage.loadContent('pages/Index', (err) => {
      if (err.code) {
        // 页面加载失败时记录错误信息
        hilog.error(DOMAIN, TAG, 'Failed to load the content. Cause: %{public}s', JSON.stringify(err));
        return;
      }
      hilog.info(DOMAIN, TAG, 'Succeeded in loading the content.');
    });
  }

  // WindowStage 销毁回调
  onWindowStageDestroy(): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onWindowStageDestroy');
  }

  // Ability 进入前台
  onForeground(): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onForeground');
  }

  // Ability 进入后台
  onBackground(): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onBackground');
  }

  // 将桌面卡片跳转携带的 formId 写入 AppStorageV2，供页面侧消费
  private consumeFormIdIntoGlobalState(want: Want, from: string): void {
    // 从 want.parameters 中读取 formId
    const formId: string = this.readStringParamFromWant(want, ROUTER_KEY_FORM_ID)
    if (formId.length === 0) {
      // 未携带 formId，说明不是从桌面卡片进入
      return
    }

    // 连接全局状态对象，用于在 Ability 与页面之间传递数据
    const st: ActiveFormState | undefined = AppStorageV2.connect(
      ActiveFormState,
      APP_STORAGE_KEY_ACTIVE_FORM,
      () => new ActiveFormState()
    )

    if (st !== undefined) {
      // 写入 formId，页面侧通常会在消费后清空
      st.formId = formId
      hilog.info(DOMAIN, TAG, '[%{public}s] saved formId=%{public}s into AppStorageV2', from, formId)
    } else {
      // AppStorage 连接失败时记录日志，便于排查上下文问题
      hilog.error(DOMAIN, TAG, 'AppStorageV2.connect failed')
    }
  }

  // 从 want.parameters 中安全读取字符串参数，避免类型异常
  private readStringParamFromWant(want: Want, key: string): string {
    if (!want.parameters) return ''
    const params: Record<string, Object> = want.parameters as Record<string, Object>
    const v: Object | undefined = params[key]
    if (v === undefined || v === null) return ''

    // 正常字符串直接返回
    if (typeof v === 'string') return v.trim()

    // 兼容 number 类型的参数
    if (typeof v === 'number') return String(v)

    // 其他类型统一忽略
    return ''
  }

  // 将通知点击携带的行情跳转参数写入 AppStorageV2
  private consumeQuoteNavIntoGlobalState(want: Want, from: string): void {
    if (!want.parameters) {
      // 没有 parameters，说明不是通知触发
      hilog.info(DOMAIN, TAG, '[%{public}s] want.parameters is empty', from)
      return
    }

    // 将 parameters 映射为本地解析结构
    const bag: QuoteNavParams = (want.parameters as Object) as QuoteNavParams

    // targetCode 是判断是否为行情跳转的关键参数
    const targetCode: string = (typeof bag.targetCode === 'string') ? bag.targetCode.trim() : ''
    if (targetCode.length === 0) {
      hilog.info(DOMAIN, TAG, '[%{public}s] no targetCode in want.parameters', from)
      return
    }

    // 其余参数允许为空，由页面侧决定默认行为
    const targetType: string = (typeof bag.targetType === 'string') ? bag.targetType.trim() : ''
    const klinePeriod: string = (typeof bag.klinePeriod === 'string') ? bag.klinePeriod.trim() : ''
    const ensureInWatchlist: boolean = (typeof bag.ensureInWatchlist === 'boolean') ? bag.ensureInWatchlist : true

    // 连接通知跳转对应的全局状态对象
    const st: ActiveQuoteNavState | undefined =
      AppStorageV2.connect(ActiveQuoteNavState, APP_STORAGE_KEY_QUOTE_NAV, () => new ActiveQuoteNavState())

    if (!st) {
      // 全局状态连接失败时直接返回
      hilog.error(DOMAIN, TAG, '[%{public}s] AppStorageV2.connect quoteNav failed', from)
      return
    }

    // 写入跳转参数
    st.targetCode = targetCode
    st.targetType = targetType
    st.klinePeriod = klinePeriod
    st.ensureInWatchlist = ensureInWatchlist

    // 递增版本号，用于页面侧判断“新的一次跳转请求”
    st.version = st.version + 1

    hilog.info(
      DOMAIN,
      TAG,
      '[%{public}s] saved quoteNav code=%{public}s type=%{public}s period=%{public}s ensure=%{public}s ver=%{public}d',
      from,
      st.targetCode,
      st.targetType,
      st.klinePeriod,
      String(st.ensureInWatchlist),
      st.version
    )
  }

  // 打印 want.parameters 中的所有 key，用于调试路由和通知传参
  private dumpWantKeys(want: Want, from: string): void {
    if (!want.parameters) {
      hilog.info(DOMAIN, TAG, '[%{public}s] want.parameters is empty', from)
      return
    }
    const params: Record<string, Object> = want.parameters as Record<string, Object>
    const keys: string[] = Object.keys(params)
    hilog.info(DOMAIN, TAG, '[%{public}s] want.parameters keys=%{public}s', from, JSON.stringify(keys))
  }
}
