@ObservedV2
export class ActiveFormState {
  @Trace formId: string = ''
}

// 建议也把 key 一起放这里，避免手滑不一致
export const APP_STORAGE_KEY_ACTIVE_FORM: string = 'active_form_state_v2'
export const ROUTER_KEY_FORM_ID: string = 'formId'