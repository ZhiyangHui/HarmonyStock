import notificationManager from '@ohos.notificationManager'
import notification from '@ohos.notification'
import wantAgent from '@ohos.wantAgent'
import { hilog } from '@kit.PerformanceAnalysisKit'
import { Haptics } from '../common/Haptics'
import { Want } from '@kit.AbilityKit'

const DOMAIN: number = 0x0000                     // hilog 使用的日志域
const TAG: string = 'HarmonyStock.Notify'         // 通知模块统一使用的日志 TAG

// 触发一条行情通知所需的最小信息集合
export interface QuoteAlertPayload {
  id: number                                     // 通知唯一 id，用于覆盖或更新
  title: string                                  // 通知标题
  text: string                                   // 通知主文案
  additionalText: string                         // 通知副文案
  bundleName: string                             // 目标应用 bundleName
  abilityName: string                            // 点击通知后要拉起的 Ability
}

// 通知点击后用于跳转行情页面的参数
export interface QuoteNavPayload {
  targetCode: string                             // 股票或指数代码
  targetType: string                             // 类型标识：stock / index
  klinePeriod: string                            // 默认展示的 K 线周期
  ensureInWatchlist: boolean                     // 是否确保加入自选
}

// 统一错误结构，仅用于日志打印 code
interface ErrorObject {
  code: string
}

// 用 class 承载 want.parameters，避免 ArkTS 对 Object literal 的限制
// 字段名需与 EntryAbility 中 QuoteNavParams 完全一致
class QuoteNavParams {
  targetCode: string = ''                        // 跳转目标代码
  targetType: string = ''                        // 跳转目标类型
  klinePeriod: string = ''                       // K 线周期
  ensureInWatchlist: boolean = true              // 是否自动加入自选

  // 将业务侧的 payload 映射为 want.parameters 可用结构
  constructor(nav: QuoteNavPayload) {
    this.targetCode = nav.targetCode
    this.targetType = nav.targetType
    this.klinePeriod = nav.klinePeriod
    this.ensureInWatchlist = nav.ensureInWatchlist
  }
}

// wantAgent.getWantAgent 所需的配置结构
// 这里避免使用 SDK 中未导出的 WantAgentInfo 类型
interface WantAgentConfig {
  wants: Array<Want>                             // 点击通知后要执行的 want 列表
  operationType: wantAgent.OperationType         // wantAgent 执行类型
  requestCode: number                            // 请求码，用于区分不同 agent
  wantAgentFlags: Array<wantAgent.WantAgentFlags>// wantAgent 行为标志
}

export class NotifyCenter {
  // 确保系统通知权限已开启，必要时主动拉起授权弹窗
  static async ensureEnabled(): Promise<boolean> {
    try {
      const enabled: boolean = await notificationManager.isNotificationEnabled()
      if (enabled) return true
      await notificationManager.requestEnableNotification()
      return await notificationManager.isNotificationEnabled()
    } catch (e) {
      // 权限检测或请求异常时记录日志
      hilog.error(DOMAIN, TAG, 'ensureEnabled err=%{public}s', JSON.stringify(e))
      return false
    }
  }

  // 发布一条行情通知，可选携带点击跳转参数
  static async publish(p: QuoteAlertPayload, nav?: QuoteNavPayload): Promise<void> {
    const ok = await NotifyCenter.ensureEnabled()
    if (!ok) return                               // 未获得通知权限时直接返回

    // 打印 nav 参数，确认通知是否携带跳转信息
    hilog.info(DOMAIN, TAG, 'publish nav=%{public}s', nav ? JSON.stringify(nav) : 'undefined')

    try {
      // 有跳转参数时将其封装为 QuoteNavParams
      const paramsObj: QuoteNavParams | undefined = nav ? new QuoteNavParams(nav) : undefined

      // 构造用于 wantAgent 的 Want 对象
      const w: Want = {
        bundleName: p.bundleName,                 // 目标应用包名
        abilityName: p.abilityName,               // 目标 Ability
        moduleName: 'entry',                      // 固定使用 entry 模块
        parameters: paramsObj
          ? ((paramsObj as Object) as Record<string, Object>)
          : undefined                              // 无跳转参数时不传 parameters
      }

      // 使用时间戳生成 requestCode，避免 wantAgent 复用旧参数
      const requestCode: number = Number(Date.now() % 2147483647)

      // wantAgent 的配置对象
      const wantAgentInfo: WantAgentConfig = {
        wants: [w],
        operationType: wantAgent.OperationType.START_ABILITY,
        requestCode: requestCode,
        wantAgentFlags: [wantAgent.WantAgentFlags.UPDATE_PRESENT_FLAG]
      }

      // 获取 wantAgent 实例，用于绑定到通知点击事件
      const agent = await wantAgent.getWantAgent(wantAgentInfo)

      // 构造系统通知请求
      const req: notificationManager.NotificationRequest = {
        id: p.id,                                 // 通知 id
        slotType: notification.SlotType.SOCIAL_COMMUNICATION,
        content: {
          notificationContentType: notificationManager.ContentType.NOTIFICATION_CONTENT_BASIC_TEXT,
          normal: {
            title: p.title,
            text: p.text,
            additionalText: p.additionalText
          }
        },
        wantAgent: agent                          // 通知点击后执行的动作
      }

      // 发布通知
      await notificationManager.publish(req)
      hilog.info(DOMAIN, TAG, 'publish ok id=%{public}d requestCode=%{public}d', req.id, requestCode)

      // 发布成功后触发一次短震动反馈
      await Haptics.vibrateShort()
    } catch (e) {
      // 捕获并记录通知发布过程中的异常
      const errObj: ErrorObject = e as ErrorObject
      hilog.error(
        DOMAIN,
        TAG,
        'publish failed code=%{public}s err=%{public}s',
        errObj.code,
        JSON.stringify(e)
      )
    }
  }
}
